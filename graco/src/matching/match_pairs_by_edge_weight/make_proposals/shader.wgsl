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

@group(0) @binding(7)
var<storage, read_write> nodes_proposal: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if has_live_nodes == 0 || index >= node_count {
        return;
    }

    let state = nodes_match_state[index];
    let status = match_state_status(state);

    if status == MATCH_STATUS_BLUE {
        var has_live_neighbour = false;
        var has_match_candidate = false;
        var best_candidate_index = 0u;
        var best_candidate_weight = 0u;

        let edges_start = nodes_edge_offset[index];

        var edges_end = edge_ref_count;

        if index < node_count - 1 {
            edges_end = nodes_edge_offset[index + 1];
        }

        for (var i = edges_start; i < edges_end; i++) {
            let other_index = nodes_edges[i];
            let edge_weight = nodes_edge_weights[i];
            let other_state = nodes_match_state[other_index];
            let other_status = match_state_status(other_state);

            if match_state_is_live(other_state) {
                has_live_neighbour = true;
            }

            if other_status == MATCH_STATUS_RED && edge_weight >= best_candidate_weight {
                has_match_candidate = true;

                best_candidate_index = other_index;
                best_candidate_weight = edge_weight;
            }
        }

        if has_match_candidate {
            atomicMax(&nodes_proposal[best_candidate_index], best_candidate_weight);

            // Also associate the index for the best candidate with this current node, to record that this node indeed
            // proposed to the candidate note. As there is never overlap between the set of proposing nodes ("blue"
            // nodes) with the set of proposed to nodes ("red nodes"), we can reuse the `nodes_proposal` buffer for this
            // purpose.
            //
            // Note that if/when WebGPU permits, this need not be an atomic operation.
            atomicStore(&nodes_proposal[index], best_candidate_index);
        }

        if !has_live_neighbour {
            nodes_match_state[index] = match_state_new_dead();
        }
    }
}
