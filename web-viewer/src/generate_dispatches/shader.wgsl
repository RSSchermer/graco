struct Draw {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

struct DrawIndexed {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(0)
var<uniform> triangle_index_count: u32;

@group(0) @binding(1)
var<uniform> node_count: u32;

@group(0) @binding(2)
var<uniform> edge_ref_count: u32;

@group(0) @binding(3)
var<storage, read_write> draw_point_triangles_dispatch: DrawIndexed;

@group(0) @binding(4)
var<storage, read_write> draw_edge_lines_dispatch: Draw;

@compute @workgroup_size(1, 1, 1)
fn main() {
    draw_point_triangles_dispatch = DrawIndexed(triangle_index_count, node_count, 0, 0, 0);
    draw_edge_lines_dispatch = Draw(edge_ref_count * 2, 1, 0, 0);
}
