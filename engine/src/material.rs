/*
  Fiddler, a UCI-compatible chess engine.
  Copyright (C) 2022 The Fiddler Authors (see AUTHORS.md file)

  Fiddler is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Fiddler is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//! Material values for each piece.
//!
//! Every piece is assigned a numeric value in centipawns (cp). Under normal
//! conditions, a centipawn is 100cp, however, the consequences of tuning have
//! yielded values for pawns slightly off of that mark.
//!
//! In traditional chess, pawns are worth 100cp, knights and bishops are worth
//! 300cp, rooks are worth 500cp, and queens are worth 900cp each. However, any
//! chess player worth their salt might tell you that bishops are a little more
//! valuable than knights. Empirically, the engine agrees.

use fiddler_base::{Board, Color, Move, Piece, Score};

/// Get the value of one piece by its type.
pub const fn value(pt: Piece) -> Score {
    match pt {
        Piece::Knight => Score::centipawns(289, 288),
        Piece::Bishop => Score::centipawns(331, 328),
        Piece::Rook => Score::centipawns(461, 456),
        Piece::Queen => Score::centipawns(962, 969),
        Piece::Pawn => Score::centipawns(103, 100), // comically, a pawn is not worth 100cp
        Piece::King => Score::centipawns(0, 0),
    }
}

/// Compute the effect that a move will have on the total quantity of material.
pub fn material_delta(b: &Board, m: Move) -> Score {
    // material only ever changes value based on captures and promotions, so
    // this is easy
    let capturee_type = if m.is_en_passant() {
        Some(Piece::Pawn)
    } else {
        b.type_at_square(m.to_square())
    };
    let mut gain = capturee_type.map_or_else(|| Score::centipawns(0, 0), value);

    if m.is_promotion() {
        // we already checked that m is a promotion, so we can trust that it has
        // a promotion
        gain += value(unsafe { m.promote_type().unwrap_unchecked() });
        gain -= value(Piece::Pawn);
    }

    // we need not put this delta in perspective, that is `Position`'s job
    gain
}

/// Evaluate a position solely by the amount of material available.
pub fn evaluate(b: &Board) -> Score {
    let mut score = Score::centipawns(0, 0);

    let white_occupancy = b[Color::White];
    let black_occupancy = b[Color::Black];

    for pt in Piece::ALL_TYPES {
        // Total the quantity of white and black pieces of this type, and
        // multiply their individual value to get the net effect on the eval.
        let pt_squares = b[pt];
        let white_diff = (white_occupancy & pt_squares).count_ones() as i8
            - (black_occupancy & pt_squares).count_ones() as i8;
        score += value(pt) * white_diff;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use fiddler_base::{
        movegen::{get_moves, NoopNominator, ALL},
        Game,
    };

    fn delta_helper(fen: &str) {
        let mut g = Game::from_fen(fen, evaluate).unwrap();
        for (m, _) in get_moves::<ALL, NoopNominator>(g.position()) {
            g.make_move(m, material_delta(g.board(), m));
            // println!("{g}");
            assert_eq!(g.position().score, evaluate(g.board()));
            g.undo().unwrap();
        }
    }

    #[test]
    fn test_delta_captures() {
        delta_helper("r1bq1b1r/ppp2kpp/2n5/3n4/2BPp3/2P5/PP3PPP/RNBQK2R b KQ d3 0 8");
    }

    #[test]
    fn test_delta_promotion() {
        // undoubling capture promotion is possible
        delta_helper("r4bkr/pPpq2pp/2n1b3/3n4/2BPp3/2P5/1P3PPP/RNBQK2R w KQ - 1 13");
    }
}
