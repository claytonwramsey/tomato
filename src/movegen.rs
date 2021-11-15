use crate::constants::{Color, BLACK};
use crate::magic::{get_bishop_attacks, get_rook_attacks, MagicTable};
use crate::moves::Move;
use crate::square::Square;
use crate::util::{opposite_color, pawn_direction, pawn_start_rank};
use crate::Bitboard;
use crate::Board;
use crate::Direction;
use crate::PieceType;

#[allow(dead_code)]
/**
 * A struct which contains all the necessary data to create moves.
 */
pub struct MoveGenerator {
    /**
     * A magic move generator.
     */
    mtable: MagicTable,
    /**
     * A bitboard of all the squares which a pawn on the given square can
     * attack.
     */
    pawn_attacks: [Bitboard; 64], //for now unused, will be used later
    /**
     * A bitboard of all the squares a king can move to if his position is the
     * index in the list.
     */
    king_moves: [Bitboard; 64],
    /**
     * A bitboard of all the squares a knight can move to if its position is
     * the index of the list.
     */
    knight_moves: [Bitboard; 64],
}

impl MoveGenerator {
    #[allow(dead_code)]
    /**
     * Load up a new MoveGenerator.
     */
    pub fn new() -> MoveGenerator {
        MoveGenerator {
            mtable: MagicTable::load(),
            pawn_attacks: create_step_attacks(&vec![Direction::NORTHEAST, Direction::NORTHWEST], 1),
            king_moves: create_step_attacks(&get_king_steps(), 1),
            knight_moves: create_step_attacks(&get_knight_steps(), 2),
        }
    }

    #[allow(dead_code)]
    /**
     * Get all the legal moves on a board.
     */
    pub fn get_moves(&self, board: &Board) -> Vec<Move> {
        let moves = self.get_pseudolegal_moves(board, board.player_to_move);
        let mut legal_moves = Vec::<Move>::new();
        for m in moves {
            let is_castle = board.is_move_castle(m);
            if !self.is_move_self_check(board, m) && !is_castle {
                legal_moves.push(m);
            }
            if is_castle {
                // TODO make castle illegal if in check or must move through
                // check
                let is_queen_castle = m.to_square().file() == 2;
                let mut is_valid = true;
                let mut king_passthru_min = 4;
                let mut king_passthru_max = 6;
                if is_queen_castle {
                    king_passthru_min = 2;
                    king_passthru_max = 4;
                }
                for file in king_passthru_min..king_passthru_max {
                    let target_sq = Square::new(m.from_square().rank(), file);
                    is_valid &= !self.is_square_attacked_by(
                        board,
                        target_sq,
                        opposite_color(board.player_to_move),
                    );
                }
                if is_valid {
                    legal_moves.push(m);
                }
            }
        }

        return legal_moves;
    }

    /**
     * Enumerate the pseudolegal moves a player of the given color would be
     * able to make if it were their turn to move.
     */
    pub fn get_pseudolegal_moves(&self, board: &Board, color: Color) -> Vec<Move> {
        let about_to_promote_bb = pawn_start_rank(opposite_color(color));
        // Number of start squares
        let num_promotion_from_squares = (board.get_type_and_color(PieceType::PAWN, color)
            & about_to_promote_bb)
            .0
            .count_ones() as usize;
        let mut normal_bitboards = Vec::with_capacity(
            board.get_color_occupancy(color).0.count_ones() as usize - num_promotion_from_squares,
        );
        let mut promotion_bitboards = Vec::with_capacity(num_promotion_from_squares);

        let pawns = board.get_type_and_color(PieceType::PAWN, color);
        let non_promoting_pawns = pawns & !about_to_promote_bb;
        let promoting_pawns = pawns & about_to_promote_bb;

        for sq in non_promoting_pawns {
            normal_bitboards.push((sq, self.pawn_moves(board, sq)));
        }
        for sq in promoting_pawns {
            promotion_bitboards.push((sq, self.pawn_moves(board, sq)));
        }

        //iterate through all the pieces of this color and enumerate their moves
        for pt in PieceType::NON_PAWN_TYPES {
            let pieces_to_move = board.get_type_and_color(pt, color);
            for sq in pieces_to_move {
                normal_bitboards.push((sq, self.sq_pseudolegal_moves(board, sq, pt)));
            }
        }

        let mut num_moves: u32 = normal_bitboards.iter().map(|x| x.1 .0.count_ones()).sum();
        num_moves += (PieceType::NUM_PROMOTE_TYPES as u32)
            * promotion_bitboards
                .iter()
                .map(|x| x.1 .0.count_ones())
                .sum::<u32>();
        let mut moves = Vec::with_capacity(num_moves as usize);
        for (from_sq, bb) in normal_bitboards {
            bitboard_to_moves(from_sq, bb, &mut moves);
        }
        for (from_sq, bb) in promotion_bitboards {
            bitboard_to_promotions(from_sq, bb, PieceType::KNIGHT, &mut moves);
            bitboard_to_promotions(from_sq, bb, PieceType::ROOK, &mut moves);
            bitboard_to_promotions(from_sq, bb, PieceType::BISHOP, &mut moves);
            bitboard_to_promotions(from_sq, bb, PieceType::QUEEN, &mut moves);
        }

        return moves;
    }

    /**
     * In a given board state, is a move illegal because it would be a
     * self-check?
     */
    pub fn is_move_self_check(&self, board: &Board, m: Move) -> bool {
        /*let mut newboard = *board;
        let player = board.color_at_square(m.from_square());
        newboard.make_move(m);
        let player_king_bb = newboard.get_type_and_color(PieceType::KING, player);
        let player_king_square = Square::from(player_king_bb);
        self.is_square_attacked_by(&newboard, player_king_square, opposite_color(player))*/

        let player = board.player_to_move;
        let player_king_bb = board.get_type_and_color(PieceType::KING, player);
        if player_king_bb == Bitboard::EMPTY {
            //if there's no king, I guess you can't check it?
            return false;
        }
        let is_king_move = player_king_bb.is_square_occupied(m.from_square());

        if is_king_move {
            //if the king is moving, we only need to know if
            return self.is_square_attacked_by(board, m.to_square(), opposite_color(player));
        }
        // The mover is not a king. Self checks can only happen by discovery.
        // Typically, only one square is emptied by moving. However, in en
        // passant, two squares are emptied. We can check the results by masking
        // out the squares which were emptied, and then seeing which attacks
        // went through using magic bitboards.

        let mut squares_emptied = Bitboard::from(m.from_square());
        if board.is_move_en_passant(m) {
            squares_emptied |= Bitboard::from(board.en_passant_square);
        }

        let occupancy = board.get_occupancy() & !squares_emptied;
        let king_square = Square::from(player_king_bb);
        let opponent = opposite_color(player);

        //The squares that a rook would see if it were in the king's square.
        let seen_rook_bb = get_rook_attacks(occupancy, king_square, &self.mtable);
        //The squares that a bishop would see if it were in the king's square.
        let seen_bishop_bb = get_bishop_attacks(occupancy, king_square, &self.mtable);

        let enemy_rook_bb = board.get_type_and_color(PieceType::ROOK, opponent);
        let enemy_queen_bb = board.get_type_and_color(PieceType::QUEEN, opponent);
        let enemey_bishop_bb = board.get_type_and_color(PieceType::BISHOP, opponent);

        //Check that the king cannot be seen by any enemy rooks, queens, or bishops.
        return (seen_rook_bb & (enemy_queen_bb | enemy_rook_bb))
            | (seen_bishop_bb & (enemy_queen_bb | enemey_bishop_bb))
            != Bitboard::EMPTY;
    }

    #[inline]
    /**
     * In a given board state, is a square attacked by the given color?
     */
    pub fn is_square_attacked_by(&self, board: &Board, sq: Square, color: Color) -> bool {
        self.get_pseudolegal_moves(board, color)
            .into_iter()
            .filter(|m| m.to_square() == sq)
            .next()
            .is_some()
    }

    #[inline]
    /**
     * Given a set of squares
     */
    pub fn are_squares_attacked_by(&self, board: &Board, squares: Bitboard, color: Color) -> bool {
        squares
            .filter(|sq| !self.is_square_attacked_by(board, *sq, color))
            .next()
            .is_some()
    }

    #[inline]
    /**
     * Enumerate all the pseudolegal moves that can be made by a given piece
     * type at the given position.
     */
    fn sq_pseudolegal_moves(&self, board: &Board, sq: Square, pt: PieceType) -> Bitboard {
        match pt {
            PieceType::PAWN => self.pawn_moves(board, sq),
            PieceType::KNIGHT => self.knight_moves(board, sq),
            PieceType::KING => self.king_moves(board, sq),
            PieceType::BISHOP => self.bishop_moves(board, sq),
            PieceType::ROOK => self.rook_moves(board, sq),
            PieceType::QUEEN => self.queen_moves(board, sq),
            //bad type gets no moves
            _ => Bitboard::EMPTY,
        }
    }

    #[inline]
    /**
     * Get the pseudolegal moves that a knight on the square `sq` could make in
     * this position. Also, haha bob seger.
     */
    fn knight_moves(&self, board: &Board, sq: Square) -> Bitboard {
        self.knight_moves[sq.0 as usize] & !board.get_color_occupancy(board.color_at_square(sq))
    }

    #[inline]
    /**
     * Get the pseudolegal moves that a king on square `sq` could make in this
     * position. Does not check if castling can be done through or out of check.
     */
    fn king_moves(&self, board: &Board, sq: Square) -> Bitboard {
        let mut moves =
            self.king_moves[sq.0 as usize] & !board.get_color_occupancy(board.color_at_square(sq));

        //castling
        let kingside_castle_passthrough_sqs = match board.player_to_move {
            BLACK => Bitboard(0x6000000000000000),
            _ => Bitboard(0x0000000000000060),
        };
        let queenside_castle_passthrough_sqs = match board.player_to_move {
            BLACK => Bitboard(0x0700000000000000),
            _ => Bitboard(0x0000000000000070),
        };

        let can_kingside_castle = board
            .castle_rights
            .is_kingside_castle_legal(board.player_to_move)
            && board.get_occupancy() & kingside_castle_passthrough_sqs == Bitboard::EMPTY;
        let can_queenside_castle = board
            .castle_rights
            .is_queenside_castle_legal(board.player_to_move)
            && board.get_occupancy() & queenside_castle_passthrough_sqs == Bitboard::EMPTY;

        if can_kingside_castle {
            moves |= Bitboard::from(Square::new(sq.rank(), 6));
        }
        if can_queenside_castle {
            moves |= Bitboard::from(Square::new(sq.rank(), 2));
        }
        return moves;
    }

    /**
     * Get the pseudolegal moves that a pawn on square `sq` could make in this
     * position.
     */
    fn pawn_moves(&self, board: &Board, sq: Square) -> Bitboard {
        let player_color = board.color_at_square(sq);
        let dir = pawn_direction(player_color);
        let start_rank = pawn_start_rank(player_color);
        let from_bb = Bitboard::from(sq);
        let occupancy = board.get_occupancy();
        let capture_sqs = [sq + dir + Direction::EAST, sq + dir + Direction::WEST];
        let opponents = board.get_color_occupancy(board.color_at_square(sq));
        let mut target_squares = Bitboard::EMPTY;
        //this will never be out of bounds because pawns don't live on promotion rank
        if !occupancy.is_square_occupied(sq + dir) {
            target_squares |= Bitboard::from(sq + dir);
            //pawn is on start rank and double-move square is not occupied
            if (start_rank & from_bb) != Bitboard::EMPTY
                && !occupancy.is_square_occupied(sq + 2 * dir)
            {
                target_squares |= Bitboard::from(sq + 2 * dir);
            }
        }
        //captures
        for capture_sq in capture_sqs {
            if capture_sq.is_inbounds() && capture_sq.chebyshev_to(sq) < 2 {
                if capture_sq == board.en_passant_square {
                    target_squares |= Bitboard::from(capture_sq);
                }
                let capture_bb = Bitboard::from(capture_sq);
                target_squares |= capture_bb & opponents;
            }
        }
        target_squares &= !board.get_color_occupancy(player_color);
        return target_squares;
    }

    #[inline]
    /**
     * Get the pseudolegal moves that a bishop on square `sq` could make in
     * this position.
     */
    fn bishop_moves(&self, board: &Board, sq: Square) -> Bitboard {
        get_bishop_attacks(board.get_occupancy(), sq, &self.mtable)
            & !board.get_color_occupancy(board.color_at_square(sq))
    }

    #[inline]
    /**
     * Get the pseudolegal moves that a rook on square `sq` could make in this
     * position.
     */
    fn rook_moves(&self, board: &Board, sq: Square) -> Bitboard {
        get_rook_attacks(board.get_occupancy(), sq, &self.mtable)
            & !board.get_color_occupancy(board.color_at_square(sq))
    }

    #[inline]
    /**
     * Get the pseudolegal moves that a queen on square `sq` could make in this
     * position.
     */
    fn queen_moves(&self, board: &Board, sq: Square) -> Bitboard {
        self.bishop_moves(board, sq) & self.rook_moves(board, sq)
    }
}

/**
 * Get the step attacks that could be made by moving in `dirs` from each point
 * in the square. Exclude the steps that travel more than `max_dist` (this
 * prevents overflow around the edges of the board).
 */
fn create_step_attacks(dirs: &[Direction], max_dist: u8) -> [Bitboard; 64] {
    let mut attacks = [Bitboard(0); 64];
    for i in 0..64usize {
        for dir in dirs {
            let start_sq = Square(i as u8);
            let target_sq = start_sq + *dir;
            if target_sq.chebyshev_to(start_sq) <= max_dist {
                attacks[i] |= Bitboard::from(target_sq);
            }
        }
    }
    return attacks;
}

#[inline]
/**
 * Given a bitboard of possible to-squares and a fixed from-square, convert
 * this to a list of `Move`s with promotion type `NO_TYPE`.
 */
fn bitboard_to_moves(from_sq: Square, bb: Bitboard, target: &mut Vec<Move>) {
    bitboard_to_promotions(from_sq, bb, PieceType::NO_TYPE, target);
}

/**
 * Given a bitboard of possible to-squares and a fixed from-square, convert
 * this to a list of `Move`s with the given promotion type and push them onto
 * the target.
 */
fn bitboard_to_promotions(
    from_sq: Square,
    bb: Bitboard,
    promote_type: PieceType,
    target: &mut Vec<Move>,
) {
    for to_sq in bb {
        target.push(Move::new(from_sq, to_sq, promote_type));
    }
}

/**
 * Get the steps a king can make.
 */
fn get_king_steps() -> Vec<Direction> {
    vec![
        Direction::NORTH,
        Direction::NORTHEAST,
        Direction::EAST,
        Direction::SOUTHEAST,
        Direction::SOUTH,
        Direction::SOUTHWEST,
        Direction::WEST,
        Direction::NORTHWEST,
    ]
}

/**
 * Get the steps a knight can make.
 */
fn get_knight_steps() -> Vec<Direction> {
    vec![
        Direction::NNW,
        Direction::NNE,
        Direction::NEE,
        Direction::SEE,
        Direction::SSE,
        Direction::SSW,
        Direction::SWW,
        Direction::NWW,
    ]
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_opening_moveset() {
        let mg = MoveGenerator::new();
        let moves = mg.get_moves(&Board::default());
        print!("{{");
        for m in moves.iter() {
            print!("{}, ", m);
        }
        print!("}}");
    }
}
