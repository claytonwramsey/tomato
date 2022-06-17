use std::{
    sync::Arc,
    thread::{spawn, JoinHandle},
    time::Instant,
};

use fiddler_base::Game;

use crate::uci::{EngineInfo, UciMessage};

use super::{
    config::SearchConfig,
    limit::SearchLimit,
    search::{search, SearchResult},
    transposition::TTable,
    SearchError,
};

#[derive(Clone, Debug)]
/// The primary search thread for an engine.
pub struct MainSearch {
    /// The configuration of the search, controlling the search parameters.
    pub config: SearchConfig,
    /// The transposition table, shared across all search threads.
    pub ttable: Arc<TTable>,
    /// The limit to the search.
    pub limit: Arc<SearchLimit>,
}

impl MainSearch {
    /// Construct a new main search with only a single search thread.
    pub fn new() -> MainSearch {
        MainSearch {
            config: SearchConfig::new(),
            ttable: Arc::new(TTable::default()),
            limit: Arc::new(SearchLimit::new()),
        }
    }

    /// Evaluate a position. The searcher will continue searching until its
    /// field `limit` marks itself as over.
    ///
    /// # Error
    ///
    /// An error will be returned according to the cases outlined in
    /// `SearchError`. Such errors are rare, and are generally either the result
    /// of an internal bug or a critical OS interrupt. However, a timeout error
    /// is most likely if the search times out before it can do any computation.
    pub fn evaluate(&self, g: &Game) -> SearchResult {
        let tic = Instant::now();
        self.ttable.age_up(2);
        let mut handles: Vec<JoinHandle<SearchResult>> = Vec::new();

        for _thread_id in 1..(self.config.n_helpers + 1) {
            let ttable_arc = self.ttable.clone();
            let limit_arc = self.limit.clone();
            let config_copy = self.config;
            let gcopy = g.clone();
            handles.push(spawn(move || {
                search(gcopy, ttable_arc, &config_copy, limit_arc, false)
            }))
        }

        // now it's our turn to think
        let mut best_result = search(
            g.clone(),
            self.ttable.clone(),
            &self.config,
            self.limit.clone(),
            true,
        );

        for handle in handles {
            let eval_result = handle.join().map_err(|_| SearchError::Join)?;

            match (best_result, eval_result) {
                // if this is our first successful thread, use its result
                (Err(_), Ok(_)) => best_result = eval_result,
                // if both were successful, use the deepest result
                (Ok(ref mut best_search), Ok(ref new_search)) => {
                    best_search.unify_with(new_search);
                }
                // error cases cause nothing to happen
                _ => (),
            };
        }
        let toc = Instant::now();
        let elapsed = toc - tic;

        if let Ok(info) = best_result {
            let nodes = self.limit.num_nodes();
            let nps = nodes * 1000 / (elapsed.as_millis() as u64);
            // inform the user
            // TODO genericize this to some kind of "UCI consumer" so that ugly
            // printouts don't go to the CLI
            print!(
                "{}",
                UciMessage::Info(&[
                    EngineInfo::Depth(info.highest_successful_depth),
                    EngineInfo::Nodes(nodes),
                    EngineInfo::NodeSpeed(nps),
                    EngineInfo::HashFull(self.ttable.fill_rate_permill())
                ])
            );
        }

        best_result
    }
}

impl Default for MainSearch {
    fn default() -> Self {
        MainSearch::new()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::max;

    use crate::pst::pst_evaluate;

    use super::*;

    /// Compare the speed of a search on a given transposition depth with its
    /// adjacent depths.
    fn transposition_speed_comparison(fen: &str, depth: u8, transposition_depth: u8, nhelpers: u8) {
        let g = Game::from_fen(fen, pst_evaluate).unwrap();
        for tdepth in max(0, transposition_depth - 1)..=(transposition_depth + 1) {
            let mut main = MainSearch::new();
            main.config.depth = depth;
            main.config.n_helpers = nhelpers;
            main.config.max_transposition_depth = tdepth;

            let tic = Instant::now();
            main.evaluate(&g).unwrap();
            let toc = Instant::now();
            println!(
                "tdepth {tdepth}: {:.3}s, hashfill {:.3}",
                (toc - tic).as_secs_f32(),
                main.ttable.fill_rate()
            );
        }
    }

    #[test]
    fn transposition_speed_fried_liver() {
        transposition_speed_comparison(
            "r1bq1b1r/ppp2kpp/2n5/3np3/2B5/8/PPPP1PPP/RNBQK2R w KQ - 0 7",
            11,
            7,
            7,
        );
    }
}
