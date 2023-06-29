#include <src/matching/match_pairs_by_edge_weight/match_state.wgsl>

@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read_write> nodes_match_state: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= count {
        return;
    }

    let state = nodes_match_state[index];
    let status = state >> 30;
    let match_index = state & 0x3FFFFFFF;

    var match_value = index;

    if status == MATCH_STATUS_MATCHED {
        match_value = match_index;
    }

    nodes_match_state[index] = match_value;
}
