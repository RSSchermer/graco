#include <src/coarsen_graph/validity.wgsl>

@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read> mapped_edges: array<u32>;

@group(0) @binding(2)
var<storage, read> validity_prefix_sum: array<u32>;

@group(0) @binding(3)
var<storage, read_write> coarse_nodes_edge_offset: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= count {
        return;
    }

    let current_offset = coarse_nodes_edge_offset[index];

    let mapped_edge = mapped_edges[current_offset];
    let validity = mapped_edge >> 30;

    var new_offset = validity_prefix_sum[current_offset] - 1;

    if validity != VALIDITY_VALID {
        new_offset += 1u;
    }

    coarse_nodes_edge_offset[index] = new_offset;
}
