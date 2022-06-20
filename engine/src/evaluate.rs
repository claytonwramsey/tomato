use std::cmp::{max, min};

use fiddler_base::{Bitboard, Board, Color, Eval, Game, Piece, Score, Move};

use crate::{pst::{pst_delta, pst_evaluate}, material::material_delta};

use super::material;

/// Mask containing ones along the A file. Bitshifting left by a number from 0 
/// through 7 will cause it to become a mask for each file.
const A_FILE_MASK: Bitboard = Bitboard::new(0x0101010101010101);

/// The value of having your own pawn doubled.
pub const DOUBLED_PAWN_VALUE: Score = (Eval::centipawns(-33), Eval::centipawns(-31));
/// The value of having a rook with no same-colored pawns in front of it which
/// are not advanced past the 3rd rank.
pub const OPEN_ROOK_VALUE: Score = (Eval::centipawns(7), Eval::centipawns(15));

/// Evaluate a leaf position on a game whose cumulative values have been 
/// computed correctly.
pub fn leaf_evaluate(g: &Game) -> Eval {
    let b = g.board();

    match g.is_over() {
        (true, Some(_)) => {
            return match b.player_to_move {
                Color::Black => Eval::mate_in(0),
                Color::White => -Eval::mate_in(0),
            }
        }
        (true, None) => {
            return Eval::DRAW;
        }
        _ => {}
    };

    let pos = g.position();
    let b = &pos.board;
    let leaf_val = leaf_rules(b);

    blend_eval(g.board(), (leaf_val.0 + pos.pst_val.0, leaf_val.1 + pos.pst_val.1))
}

/// Compute the change in scoring that a move made on a board will cause. Used 
/// in tandem with `leaf_evaluate()`.
pub fn value_delta(b: &Board, m: Move) -> Score {
    let pst_delta = pst_delta(b, m);
    let material_change = material_delta(b, m);

    (pst_delta.0 + material_change.0, pst_delta.1 + material_change.1)
}

/// Compute a static, cumulative-invariant evaluation of a position. It is much 
/// faster in search to use cumulative evaluation, but this should be used when 
/// importing positions. Static evaluation will not include the leaf rules (such 
/// as number of doubled pawns), as this will be handled by `leaf_evaluate` at 
/// the end of the search tree.
pub fn static_evaluate(b: &Board) -> Score {
    let mut value = material::evaluate(b);

    let pst_value = pst_evaluate(b);
    
    value.0 += pst_value.0;
    value.1 += pst_value.1;

    // do not include leaf rules in static evaluation

    value
}

/// Get the score gained from evaluations that are only performed at the leaf.
fn leaf_rules(b: &Board) -> Score {
    // Add losses due to doubled pawns
    let ndoubled = net_doubled_pawns(b);
    let mut mg_eval = DOUBLED_PAWN_VALUE.0 * ndoubled;
    let mut eg_eval = DOUBLED_PAWN_VALUE.1 * ndoubled;

    // Add gains from open rooks
    let nopen = net_open_rooks(b);
    mg_eval += OPEN_ROOK_VALUE.0 * nopen;
    eg_eval += OPEN_ROOK_VALUE.1 * nopen;

    (mg_eval, eg_eval)
}

/// Count the number of "open" rooks (i.e., those which are not blocked by
/// unadvanced pawns) in a position. The number is a net value, so it will be
/// negative if Black has more open rooks than White.
pub fn net_open_rooks(b: &Board) -> i8 {
    // Mask for pawns which are above rank 3 (i.e. on the white half of the
    // board).
    const BELOW_RANK3: Bitboard = Bitboard::new(0xFFFFFFFF);
    // Mask for pawns which are on the black half of the board
    const ABOVE_RANK3: Bitboard = Bitboard::new(0x00000000FFFFFFFF);
    let mut net_open_rooks = 0i8;
    let rooks = b[Piece::Rook];
    let pawns = b[Piece::Pawn];
    let white = b[Color::White];
    let black = b[Color::Black];

    // count white rooks
    for wrook_sq in rooks & white {
        if wrook_sq.rank() >= 3 {
            net_open_rooks += 1;
            continue;
        }
        let pawns_in_col = (pawns & white) & (A_FILE_MASK << wrook_sq.file());
        let important_pawns = BELOW_RANK3 & pawns_in_col;
        // check that the forward-most pawn of the important pawns is in front
        // of or behind the rook
        if important_pawns.leading_zeros() > (63 - (wrook_sq as u32)) {
            // all the important pawns are behind the rook
            net_open_rooks += 1;
        }
    }

    // count black rooks
    for brook_sq in rooks & black {
        if brook_sq.rank() <= 4 {
            net_open_rooks -= 1;
            continue;
        }
        let pawns_in_col = (pawns & black) & (A_FILE_MASK << brook_sq.file());
        let important_pawns = ABOVE_RANK3 & pawns_in_col;
        // check that the lowest-rank pawn that could block the rook is behind
        // the rook
        if important_pawns.trailing_zeros() > brook_sq as u32 {
            net_open_rooks -= 1;
        }
    }

    net_open_rooks
}

/// Count the number of doubled pawns, in net. For instance, if White had 1 
/// doubled pawn, and Black had 2, this function would return -1.
pub fn net_doubled_pawns(b: &Board) -> i8 {
    let white_occupancy = b[Color::White];
    let pawns = b[Piece::Pawn];
    let mut npawns: i8 = 0;
    let mut col_mask = Bitboard::new(0x0101010101010101);
    for _ in 0..8 {
        let col_pawns = pawns & col_mask;

        // all ones on the A column, shifted left by the col
        let num_black_doubled_pawns = match ((!white_occupancy) & col_pawns).count_ones() {
            0 => 0,
            x => x as i8 - 1,
        };
        let num_white_doubled_pawns = match (white_occupancy & col_pawns).count_ones() {
            0 => 0,
            x => x as i8 - 1,
        };

        npawns -= num_black_doubled_pawns;
        npawns += num_white_doubled_pawns;

        col_mask <<= 1;
    }

    npawns
}

/// Get a blending float describing the current phase of the game. Will range
/// from 0 (full endgame) to 1 (full midgame).
pub fn phase_of(b: &Board) -> f32 {
    const MG_LIMIT: Eval = Eval::centipawns(2500);
    const EG_LIMIT: Eval = Eval::centipawns(1400);
    // amount of non-pawn material in the board, under midgame values
    let mg_npm = {
        let mut total = Eval::DRAW;
        for pt in Piece::NON_PAWN_TYPES {
            total += material::value(pt).0 * b[pt].count_ones();
        }
        total
    };
    let bounded_npm = max(MG_LIMIT, min(EG_LIMIT, mg_npm));

    (bounded_npm - EG_LIMIT).float_val() / (MG_LIMIT - EG_LIMIT).float_val()
}

#[inline(always)]
/// Blend the evaluation of a position between the midgame and endgame.
pub fn blend_eval(b: &Board, score: Score) -> Eval {
    phase_blend(phase_of(b), score)
}

/// Blend a score based on a phase.
pub fn phase_blend(phase: f32, score: Score) -> Eval {
    score.0 * phase + score.1 * (1. - phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fiddler_base::movegen::{get_moves, NoopNominator, ALL};

    fn delta_helper(fen: &str) {
        let mut g = Game::from_fen(fen, static_evaluate).unwrap();
        for (m, _) in get_moves::<ALL, NoopNominator>(g.position()) {
            g.make_move(m, value_delta(g.board(), m));
            // println!("{g}");
            assert_eq!(static_evaluate(g.board()), g.position().pst_val);
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