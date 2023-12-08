#include <src/matching/match_pairs_by_edge_weight/match_state.wgsl>

@group(0) @binding(0)
var<uniform> node_count: u32;

@group(0) @binding(1)
var<uniform> edge_ref_count: u32;

@group(0) @binding(2)
var<uniform> has_live_nodes: u32;

@group(0) @binding(3)
var<storage, read_write> nodes_match_state: array<MatchState>;

@group(0) @binding(4)
var<storage, read> nodes_edge_offset: array<u32>;

@group(0) @binding(5)
var<storage, read> nodes_edges: array<u32>;

@group(0) @binding(6)
var<storage, read> nodes_edge_weights: array<u32>;

// Note that this is a dual-purpose buffer: for proposed-to nodes ("red" nodes), this buffer contains the weight of the
// winning proposal (the proposal with the heighest weight); for proposing nodes ("blue" nodes), this buffer contains
// the index of the proposed-to node (for nodes that are neither "red" nor "blue" - "dead" or "matched" nodes - the
// entries are never used). This is possible because there is no overlap between "red" and "blue" nodes (a node cannot
// be both "red" and "blue"), thus allowing us to reduce memory usage.
@group(0) @binding(7)
var<storage, read> nodes_proposal: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if has_live_nodes == 0 || index >= node_count {
        return;
    }

    let state = nodes_match_state[index];
    let status = match_state_status(state);
    let proposal = nodes_proposal[index];

    if status == MATCH_STATUS_RED && proposal != 0 {
        let proposal_weight = proposal;

        let edges_start = nodes_edge_offset[index];

        var edges_end = edge_ref_count;

        if index < node_count - 1 {
            edges_end = nodes_edge_offset[index + 1];
        }

        // Loop over all adjacent nodes to find the proposing node ("blue" node) with the proposal's edge weight. Also
        // verify that the proposing node did in fact propose to this current node, as there may be multiple adjacent
        // nodes with the same edge weight. If we find such a node, then create a match.
        for (var i = edges_start; i < edges_end; i++) {
            let other_index = nodes_edges[i];
            let edge_weight = nodes_edge_weights[i];
            let other_proposal_target_index = nodes_proposal[other_index];
            let other_state = nodes_match_state[other_index];
            let other_status = match_state_status(other_state);

            if other_status == MATCH_STATUS_BLUE && edge_weight == proposal_weight && other_proposal_target_index == index {
                let match_index = min(index, other_index);
                let new_state = match_state_new_matched(match_index);

                nodes_match_state[index] = new_state;
                nodes_match_state[other_index] = new_state;

                break;
            }
        }
    }
}
