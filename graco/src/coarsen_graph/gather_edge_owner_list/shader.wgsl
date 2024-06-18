@group(0) @binding(0)
var<uniform> fine_node_count: u32;

@group(0) @binding(1)
var<uniform> fine_edge_count: u32;

@group(0) @binding(2)
var<storage, read> fine_nodes_edge_offset: array<u32>;

@group(0) @binding(3)
var<storage, read> fine_nodes_mapping: array<u32>;

@group(0) @binding(4)
var<storage, read_write> coarsened_edge_owner_list: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index < fine_node_count {
        let start = fine_nodes_edge_offset[index];

        var end = fine_edge_count;

        if index < fine_node_count - 1 {
            end = fine_nodes_edge_offset[index + 1];
        }

        let owner_index = fine_nodes_mapping[index];

        for (var i = start; i < end; i += 1u) {
            coarsened_edge_owner_list[i] = owner_index;
        }
    }
}
