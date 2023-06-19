#include <src/matching/match_pairs_by_edge_weight/match_state.wgsl>

@group(0) @binding(0)
var<uniform> has_live_nodes: u32;

@group(0) @binding(1)
var<storage, read_write> nodes_match_state: array<MatchState>;

@group(0) @binding(2)
var<storage, read> nodes_edge_offset: array<u32>;

@group(0) @binding(3)
var<storage, read> nodes_edges: array<u32>;

@group(0) @binding(4)
var<storage, read> nodes_edge_weights: array<f32>;

@group(0) @binding(5)
var<storage, read> nodes_proposal: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if has_live_nodes == 0 || index >= arrayLength(&nodes_match_state) {
        return;
    }

    let state = nodes_match_state[index];
    let status = match_state_status(state);
    let proposal = nodes_proposal[index];

    if status == MATCH_STATUS_RED && proposal != 0 {
        let proposal_weight = bitcast<f32>(proposal);

        let edges_start = nodes_edge_offset[index];

        var edges_end = arrayLength(&nodes_edges);

        if index < arrayLength(&nodes_edge_offset) - 1 {
            edges_end = nodes_edge_offset[index + 1];
        }

        for (var i = edges_start; i < edges_end; i++) {
            let other_index = nodes_edges[i];
            let edge_weight = nodes_edge_weights[i];
            let other_state = nodes_match_state[other_index];
            let other_status = match_state_status(other_state);

            if other_status == MATCH_STATUS_BLUE && edge_weight == proposal_weight {
                let match_index = min(index, other_index);
                let new_state = match_state_new_matched(match_index);

                nodes_match_state[index] = new_state;
                nodes_match_state[other_index] = new_state;

                break;
            }
        }
    }
}
