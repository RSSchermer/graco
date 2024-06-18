#include <src/coarsen_graph/validity.wgsl>

@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read> owner_nodes: array<u32>;

@group(0) @binding(2)
var<storage, read_write> mapped_edges: array<u32>;

@group(0) @binding(3)
var<storage, read_write> validity: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= count {
        return;
    }

    let mapped_edge = mapped_edges[index] & 0x3FFFFFFF;
    let owner_node = owner_nodes[index];

    let is_self_reference = mapped_edge == owner_node;

    let is_duplicate = index > 0
        && mapped_edge == (mapped_edges[index - 1] & 0x3FFFFFFF)
        && owner_node == owner_nodes[index - 1];

    var validity_state: u32;

    if is_self_reference {
        validity_state = VALIDITY_INVALID_SELF_REFERENCE;
    } else if is_duplicate {
        validity_state = VALIDITY_INVALID_DUPLICATE;
    } else {
        validity_state = VALIDITY_VALID;
    }

    if validity_state == VALIDITY_VALID {
        validity[index] = 1u;
    } else {
        validity[index] = 0u;
    }

    // Note: we write and read from this buffer concurrently during this pass, but we always write back the exact same
    // value to the 30 least significant bits, and we only make use of the 30 least significant bits when we did a read.
    // For a word-sized and aligned read, there should never be any read/write tearing, so we should be good to do this
    // without using atomics.
    mapped_edges[index] = mapped_edge | (validity_state << 30);
}
