/*
  Tomato, a UCI-compatible chess engine.
  Copyright (C) 2022 Clayton Ramsey.

  Tomato is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Tomato is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//! Thread management and synchronization.
//!
//! This is the meat of parallelism in the engine: a `MainSearch` is responsible or corralling all
//! the threads and getting them to work together and on time.
//! The main search also collects all of the output from each individual search and composes it into
//! a single easily-used structure for consumption in the main process.

use std::time::Instant;

use crate::base::game::Game;

use super::{
    evaluate::Eval,
    uci::{EngineInfo, Message},
};

use super::{
    limit::SearchLimit,
    search::{search, SearchResult},
    transposition::TTable,
    SearchError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Configuration options for a search.
pub struct SearchConfig {
    /// The depth at which this algorithm will evaluate a position.
    pub depth: u8,
    /// The number of moves at each layer which will be searched to a full depth, as opposed to a
    /// lower-than-target depth.
    pub num_early_moves: usize,
    /// The number of nodes which have to be searched before it is worthwhile to update the search
    /// limit with this information.
    pub limit_update_increment: u64,
}

impl SearchConfig {
    #[must_use]
    pub fn new() -> SearchConfig {
        SearchConfig {
            depth: 10,
            num_early_moves: 4,
            limit_update_increment: 100,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> SearchConfig {
        SearchConfig::new()
    }
}

#[derive(Debug)]
/// A convenient wrapper for searching a game.
pub struct MainSearch {
    /// The configuration of the search, controlling the search parameters.
    pub config: SearchConfig,
    /// The transposition table.
    pub ttable: TTable,
    /// The limit to the search.
    pub limit: SearchLimit,
}

impl MainSearch {
    #[must_use]
    /// Construct a new main search with only a single search thread.
    pub fn new() -> MainSearch {
        MainSearch {
            config: SearchConfig::new(),
            ttable: TTable::with_size(0),
            limit: SearchLimit::new(),
        }
    }

    /// Evaluate a position.
    /// The searcher will continue searching until its field `limit` marks itself as over.
    ///
    /// # Errors
    ///
    /// An error will be returned according to the cases outlined in `SearchError`.
    /// Such errors are rare, and are generally either the result of an internal bug or a critical
    /// OS interrupt.
    /// However, a timeout error is most likely if the search times out before it can do any
    /// computation.
    pub fn evaluate(&self, g: &Game) -> SearchResult {
        let tic = Instant::now();
        let mut best_result = Err(SearchError::Timeout);

        // The previous iteration's evaluation, used for windowing
        let mut prev_eval = None;
        for depth in 1..=self.config.depth {
            // now it's our turn to think
            let sub_result = self.aspiration_search(g, depth, prev_eval);

            if sub_result.is_ok() {
                // update best result and inform GUI
                best_result = sub_result;
                let elapsed = tic.elapsed();
                if let Ok(ref best_info) = best_result {
                    prev_eval = Some(best_info.eval);
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        println!(
                            "{}",
                            Message::Info(&[
                                EngineInfo::Depth(best_info.depth),
                                EngineInfo::Score {
                                    eval: best_info.eval,
                                    is_lower_bound: false,
                                    is_upper_bound: false
                                },
                                EngineInfo::Nodes(best_info.num_nodes_evaluated),
                                EngineInfo::NodeSpeed(
                                    1000 * best_info.num_nodes_evaluated
                                        / (elapsed.as_millis() + 1) as u64
                                ),
                                EngineInfo::Time(elapsed),
                                EngineInfo::Pv(&best_info.pv),
                                EngineInfo::HashFull(self.ttable.fill_rate_permill()),
                                EngineInfo::SelDepth(best_info.selective_depth),
                            ])
                        );
                    }
                }
            }
        }

        if let Ok(ref mut info) = best_result {
            // normalize evaluation to be in absolute terms
            info.eval = info.eval.in_perspective(g.board().player);
        }
        best_result
    }

    fn aspiration_search(&self, g: &Game, depth: u8, prev_eval: Option<Eval>) -> SearchResult {
        if let Some(ev) = prev_eval {
            // we have a previous score we can use to window this search
            let (alpha, beta) = if ev.is_mate() {
                if ev < Eval::DRAW {
                    // we are getting mated.
                    // search for ways we can get mated faster
                    (Eval::MIN, ev + Eval::centipawns(1))
                } else {
                    // we are matting, search for faster wins
                    (ev - Eval::centipawns(1), Eval::MAX)
                }
            } else {
                match depth & 0x1u8 {
                    // even depth means that we expect the evaluation to decrease
                    0 => (ev - Eval::centipawns(100), ev + Eval::centipawns(10)),
                    // odd depth means that we expect the evaluation to increase
                    1 => (ev - Eval::centipawns(10), ev + Eval::centipawns(100)),
                    _ => unreachable!(),
                }
            };
            let window_result = search(
                g.clone(),
                depth,
                &self.ttable,
                &self.config,
                &self.limit,
                alpha,
                beta,
            );

            if let Ok(ref res) = window_result {
                if alpha < res.eval && res.eval < beta {
                    return window_result;
                }
            }
        }

        search(
            g.clone(),
            depth,
            &self.ttable,
            &self.config,
            &self.limit,
            Eval::MIN,
            Eval::MAX,
        )
    }
}

impl Default for MainSearch {
    fn default() -> Self {
        MainSearch::new()
    }
}

#[cfg(any(test, bench))]
mod tests {

    use crate::base::movegen::is_legal;

    use super::*;

    fn search_helper(fen: &str, depth: u8) {
        let mut g = Game::from_fen(fen).unwrap();
        let mut main = MainSearch::new();
        main.config.depth = depth;
        main.ttable.resize(1000);
        let info = main.evaluate(&g).unwrap();
        for m in info.pv {
            assert!(is_legal(m, g.board()));
            g.make_move(m);
        }
    }

    #[test]
    fn search_opening() {
        search_helper(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            6,
        );
    }

    #[test]
    fn search_fried_liver() {
        search_helper(
            "r1bq1b1r/ppp2kpp/2n5/3np3/2B5/8/PPPP1PPP/RNBQK2R w KQ - 0 7",
            6,
        );
    }
}
