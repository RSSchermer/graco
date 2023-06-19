#pragma once

const MATCH_STATUS_BLUE = 0u;
const MATCH_STATUS_RED = 1u;
const MATCH_STATUS_DEAD = 2u;
const MATCH_STATUS_MATCHED = 3u;

struct MatchState {
    packed_data: u32
}

fn match_state_new_blue() -> MatchState {
    return MatchState(0);
}

fn match_state_new_red() -> MatchState {
    let packed_data = MATCH_STATUS_RED << 30;

    return MatchState(packed_data);
}

fn match_state_new_dead() -> MatchState {
    let packed_data = MATCH_STATUS_DEAD << 30;

    return MatchState(packed_data);
}

fn match_state_new_matched(index: u32) -> MatchState {
    let packed_data = index | (MATCH_STATUS_MATCHED << 30);

    return MatchState(packed_data);
}

fn match_state_status(match_state: MatchState) -> u32 {
    return match_state.packed_data >> 30;
}

fn match_state_is_live(match_state: MatchState) -> bool {
    let status = match_state_status(match_state);

    return status == MATCH_STATUS_BLUE || status == MATCH_STATUS_RED;
}

fn match_state_match_index(match_state: MatchState) -> u32 {
    return match_state.packed_data & 0x3FFFFFFF;
}
