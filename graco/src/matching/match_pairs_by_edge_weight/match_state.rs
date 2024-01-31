use std::fmt;

use bytemuck::Zeroable;
use empa::abi;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u32)]
pub enum MatchStatus {
    Blue = 0,
    Red = 1,
    Dead = 2,
    Matched = 3,
}

#[derive(abi::Sized, Clone, Copy, PartialEq, Zeroable)]
#[repr(C)]
pub struct MatchState {
    packed_data: u32,
}

impl fmt::Debug for MatchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = self.packed_data >> 30;

        let status = match status {
            0 => MatchStatus::Blue,
            1 => MatchStatus::Red,
            2 => MatchStatus::Dead,
            3 => MatchStatus::Matched,
            _ => unreachable!(),
        };

        let match_index = self.packed_data & 0x3FFFFFFF;

        f.debug_struct("MatchState")
            .field("status", &status)
            .field("match_index", &match_index)
            .finish()
    }
}
