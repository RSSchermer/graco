@group(0) @binding(0)
var<uniform> node_count: u32;

@group(0) @binding(1)
var<uniform> edge_ref_count: u32;

@group(0) @binding(2)
var<storage, read> nodes_edge_offset: array<u32>;

@group(0) @binding(3)
var<storage, read> nodes_edges: array<u32>;

@group(0) @binding(4)
var<storage, read> nodes_position: array<vec2<f32>>;

@group(0) @binding(5)
var<storage, read_write> nodes_edge_weights: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= node_count {
        return;
    }

    let start = nodes_edge_offset[index];

    var end = edge_ref_count;

    if index < node_count - 1 {
        end = nodes_edge_offset[index + 1];
    }

    let pos_a = nodes_position[index];

    for(var i = start; i < end; i += 1u) {
        let target_index = nodes_edges[i];
        let pos_b = nodes_position[target_index];

        let dist = distance(pos_a, pos_b);
        let dist_inv = 1.0 / dist;

        nodes_edge_weights[i] = u32(dist_inv * 1000.0);
    }
}
