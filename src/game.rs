use crate::board::Board;
use crate::r#move::Move;
use crate::movegen::MoveGenerator;

use std::collections::HashMap;
use std::default::Default;

/**
 * A struct containing game information, which unlike a `Board`, knows about 
 * its history and can do things like repetition timing.
 */
pub struct Game {
    /**
     * The last element in `history` is the current state of the board. The 
     * first element should be the starting position of the game, and in 
     * between are sequential board states from the entire game.
     */
    history: Vec<Board>,
    /**
     * `moves` is the list, in order, of all moves made in the game. They   
     * should all be valid moves. The length of `moves` should always be one 
     * less than the length of `history`.
     */
    moves: Vec<Move>,
    /**
     * Stores the number of times a position has been reached in the course of 
     * this game. It is used for three-move-rule draws.
     */
    repetitions: HashMap<Board, u8>,
    //TODO figure out how to implement fifty-move rule here.
}

impl Game {
    
    #[allow(dead_code)]
    /**
     * Empty out the history of this game completely, but leave the original 
     * start state of the board.
     */
    pub fn clear(&mut self) {
        self.history.truncate(1);
        let start_board = self.history[0];
        self.moves.clear();
        self.repetitions.clear();
        //since we cleared this, or_insert will always be called
        self.repetitions.entry(start_board).or_insert(1);
    }
    
    #[allow(dead_code)]
    /**
     * Make a move, assuming said move is illegal. If the history is empty 
     * (this should never happen if normal operations occurred), the move will 
     * be made from the default state of a `Board`.
     */
    pub fn make_move(&mut self, m: Move) {
        let mut newboard = match self.history.last() {
            Some(b) => *b,
            None => Board::default(),
        };
        
        newboard.make_move(m);

        let num_reps = self.repetitions.entry(newboard).or_insert(0);
        *num_reps += 1;
        self.history.push(newboard);
        self.moves.push(m);
    }
    
    #[allow(dead_code)]
    /**
     * Attempt to play a move, which may or may not be legal. If the move is 
     * legal, the move will be executed and the state will change, then 
     * `Ok(())` will be returned. If not, an `Err` will be returned to inform 
     * you that the move is illegal, and no state will be changed.
     */
    pub fn try_move(&mut self, mgen: &MoveGenerator, m: Move) -> Result<(), &'static str>{
        let prev_board = match self.history.last() {
            Some(b) => *b,
            None => Board::default(),
        };

        if mgen.get_moves(&prev_board).contains(&m) {
            self.make_move(m);
            return Ok(());
        }
        else {
            return Err("illegal move given!");
        }

    }
    
    #[allow(dead_code)]
    /**
     * Undo the most recent move. The return will be `Ok` if there are moves 
     * left to undo, with the internal value being the move that was undone, 
     * and `Err` if there are no moves to undo.
     */
    pub fn undo(&mut self) -> Result<Move, &'static str> {
        
        let move_removed = match self.moves.pop() {
            Some(m) => m,
            None => return Err("no moves to remove"),
        };
        let state_removed = match self.history.pop() {
            Some(b) => b,
            None => return Err("no boards in history"),
        };
        let num_reps = self.repetitions.entry(state_removed).or_insert(1);
        *num_reps -= 1;
        if *num_reps <= 0 {
            self.repetitions.remove(&state_removed);
        }

        Ok(move_removed)
    }
    
    #[allow(dead_code)]
    /**
     * Undo a set number of moves. Returns an Err if you attempt to remove too 
     * many moves (and will not undo anything if that is the case).
     */
    pub fn undo_n(&mut self, nmoves: usize) -> Result<(), &'static str> {
        if nmoves > self.moves.len() {
            return Err("attempted to remove more moves than are in history");
        }
        for _ in 0..nmoves {
            self.undo()?;
        }
        Ok(())
    }

    #[inline]
    #[allow(dead_code)]
    /**
     * Get the current state of the game as a board. Will return a default 
     * Board if the history is empty (but this should never happen).
     */
    pub fn get_board(&self) -> Board {
        return *self.history.last().unwrap_or(&Board::default());
    }
}

impl Default for Game {
    fn default() -> Self {
        Game {
            history: vec![Board::default()],
            moves: Vec::new(),
            repetitions: HashMap::from([
                (Board::default(), 1),
            ]),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::r#move::Move;
    use crate::square::*;
    use crate::piece::NO_TYPE;
    use crate::board;
    use super::*;

    #[test]
    fn test_play_e4() {
        let mut g = Game::default();
        let m = Move::new(E2, E4, NO_TYPE);
        let old_board = g.get_board();
        g.make_move(Move::new(E2, E4, NO_TYPE));
        let new_board = g.get_board();
        board::tests::test_move_result_helper(old_board, new_board, m);
    }

    #[test]
    fn test_undo_move() {
        let mut g = Game::default();
        let m = Move::new(E2, E4, NO_TYPE);
        g.make_move(m);
        assert_eq!(g.undo(), Ok(m));
        assert_eq!(g.get_board(), Board::default());
    }
}