#include <src/coarsen_graph/validity.wgsl>

@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read> mapped_edges: array<u32>;

@group(0) @binding(2)
var<storage, read> mapped_edge_weights: array<u32>;

@group(0) @binding(3)
var<storage, read> validity_prefix_sum: array<u32>;

@group(0) @binding(4)
var<storage, read_write> coarse_nodes_edge_weights: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= count {
        return;
    }

    let mapped_edge = mapped_edges[index];

    let edge_ref = mapped_edge & 0x3FFFFFFF;
    let validity = mapped_edge >> 30;

    if validity != VALIDITY_INVALID_SELF_REFERENCE {
        // Note that due to the construction of the inclusive validity prefix-sum, `validity_prefix_sum[index]` is
        // always greater than 0 not just for valid edges, but also for invalid-but-not-self-referencing duplicate
        let dest_index = validity_prefix_sum[index] - 1;

        atomicAdd(&coarse_nodes_edge_weights[dest_index], mapped_edge_weights[index]);
    }
}
