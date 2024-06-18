struct EdgeVertex {
    position: array<f32, 2>,
    color: array<f32, 3>,
}

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
var<storage, read> nodes_matching: array<u32>;

@group(0) @binding(6)
var<storage, read_write> edge_vertices: array<EdgeVertex>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= node_count {
        return;
    }

    let edges_start = nodes_edge_offset[index];

    var edges_end = edge_ref_count;

    if index < node_count - 1 {
        edges_end = nodes_edge_offset[index + 1];
    }

    let match_a = nodes_matching[index];
    let start_pos = nodes_position[index];

    for (var i = edges_start; i < edges_end; i += 1u) {
        let other_index = nodes_edges[i];
        let end_pos = nodes_position[other_index];

        let offset = i * 2;

        edge_vertices[offset].position = array(start_pos.x, start_pos.y);
        edge_vertices[offset + 1].position = array(end_pos.x, end_pos.y);

        let match_b = nodes_matching[other_index];

        var color = array(0.0, 0.0, 0.0);

        if match_a != 0xFFFFFFFFu && match_a == match_b {
            color = array(1.0, 0.0, 0.0);
        }

        edge_vertices[offset].color = color;
        edge_vertices[offset + 1].color = color;
    }
}
