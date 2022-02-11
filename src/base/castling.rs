use crate::base::Color;

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
///
/// A simple struct to store a piece's castling rights.
/// The internal bits are used to represent castling rights.
/// From MSB to LSB:
/// * 4 unused bits
/// * Black queenside castling
/// * Black kingside castling
/// * White queenside castling
/// * White kingside castling
///
pub struct CastleRights(pub u8);

impl CastleRights {
    pub const ALL_RIGHTS: CastleRights = CastleRights(15);
    pub const NO_RIGHTS: CastleRights = CastleRights(0);
    pub const WHITERIGHTS: CastleRights = CastleRights(3);
    pub const BLACKRIGHTS: CastleRights = CastleRights(12);

    ///
    /// Create a `CastleRights` for kingside castling on one side.
    ///
    #[inline]
    pub const fn king_castle(color: Color) -> CastleRights {
        match color {
            Color::White => CastleRights(1),
            Color::Black => CastleRights(4),
        }
    }

    ///
    /// Create a `CastleRights` for queenside castling on one side.
    ///
    #[inline]
    pub const fn queen_castle(color: Color) -> CastleRights {
        match color {
            Color::White => CastleRights(2),
            Color::Black => CastleRights(8),
        }
    }

    ///
    /// Get the full rights for one color.
    ///
    pub const fn color_rights(color: Color) -> CastleRights {
        match color {
            Color::White => CastleRights(3),
            Color::Black => CastleRights(12),
        }
    }

    #[inline]
    ///
    /// Can the given color legally castle kingside?
    ///
    pub fn is_kingside_castle_legal(&self, color: Color) -> bool {
        *self & CastleRights::king_castle(color) != CastleRights::NO_RIGHTS
    }

    #[inline]
    ///
    /// Can the given color legally castle kingside?
    ///
    pub fn is_queenside_castle_legal(&self, color: Color) -> bool {
        *self & CastleRights::queen_castle(color) != CastleRights::NO_RIGHTS
    }
}

impl BitOr<CastleRights> for CastleRights {
    type Output = CastleRights;
    #[inline]
    fn bitor(self, other: CastleRights) -> CastleRights {
        CastleRights(self.0 | other.0)
    }
}

impl BitOrAssign<CastleRights> for CastleRights {
    #[inline]
    fn bitor_assign(&mut self, other: CastleRights) {
        self.0 |= other.0;
    }
}

impl BitAnd<CastleRights> for CastleRights {
    type Output = CastleRights;
    #[inline]
    fn bitand(self, other: CastleRights) -> CastleRights {
        CastleRights(self.0 & other.0)
    }
}

impl BitAndAssign<CastleRights> for CastleRights {
    #[inline]
    fn bitand_assign(&mut self, other: CastleRights) {
        self.0 &= other.0;
    }
}

impl Not for CastleRights {
    type Output = CastleRights;
    #[inline]
    fn not(self) -> CastleRights {
        CastleRights(self.0 ^ 15)
    }
}
