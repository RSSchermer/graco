@group(0) @binding(0)
var<uniform> child_level_node_count: u32;

@group(0) @binding(1)
var<uniform> parent_level_node_count: u32;

@group(0) @binding(2)
var<storage, read> parent_level_positions: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read> coarse_nodes_mapping_offset: array<u32>;

@group(0) @binding(4)
var<storage, read> coarse_nodes_mapping: array<u32>;

@group(0) @binding(5)
var<storage, read_write> child_level_positions: array<vec2<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index >= child_level_node_count {
        return;
    }

    let start = coarse_nodes_mapping_offset[index];

    var end = parent_level_node_count;

    if index < child_level_node_count - 1 {
        end = coarse_nodes_mapping_offset[index + 1];
    }

    var position_sum = vec2(0.0, 0.0);

    for(var i = start; i < end; i += 1u) {
        let mapping = coarse_nodes_mapping[i];

        position_sum += parent_level_positions[mapping];
    }

    let size = end - start;

    child_level_positions[index] = position_sum / f32(size);
}
