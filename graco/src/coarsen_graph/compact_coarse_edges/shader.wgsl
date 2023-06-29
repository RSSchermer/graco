#include <src/coarsen_graph/validity.wgsl>

@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read> mapped_edges: array<u32>;

@group(0) @binding(2)
var<storage, read> validity_prefix_sum: array<u32>;

@group(0) @binding(3)
var<storage, read_write> coarse_nodes_edges: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= count {
        return;
    }

    let mapped_edge = mapped_edges[index];

    let edge_ref = mapped_edge & 0x3FFFFFFF;
    let validity = mapped_edge >> 30;

    if validity == VALIDITY_VALID {
        let new_index = validity_prefix_sum[index] - 1;

        coarse_nodes_edges[new_index] = edge_ref;
    }
}
