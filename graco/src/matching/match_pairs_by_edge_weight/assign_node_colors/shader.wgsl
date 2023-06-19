#include <src/matching/match_pairs_by_edge_weight/match_state.wgsl>

@group(0) @binding(0)
var<uniform> prng_seed: u32;

@group(0) @binding(1)
var<storage, read_write> nodes_match_state: array<MatchState>;

@group(0) @binding(2)
var<storage, read_write> has_live_nodes: u32;

// Based on Schechter et al. Evolving Sub-Grid Turbulence for Smoke Animation.
// https://www.cs.ubc.ca/~rbridson/docs/schechter-sca08-turbulence.pdf
fn prng_hash(state: u32) -> u32 {
    var s = state;

    s ^= 2747636419u;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;

    return s;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= arrayLength(&nodes_match_state) {
        return;
    }

    let state = nodes_match_state[index];

    if match_state_is_live(state) {
        let v = prng_hash(prng_seed + index);

        var new_state: MatchState;

        // Under the assumption that our prng_hash function achieves a (reasonably) uniform distribution accross the
        // 32 bit numbers, then this threshold corresponds to an approximately 0.53406 chance of assigning a node the
        // "blue" color. For the motivation of this probability, see Auer et al. A GPU Algorithm for Greedy Graph
        // Matching. https://webspace.science.uu.nl/~bisse101/Articles/match12.pdf
        if v < 2293770234u {
            new_state = match_state_new_blue();
        } else {
            new_state = match_state_new_red();
        }

        nodes_match_state[index] = new_state;

        has_live_nodes = 1;
    }
}
