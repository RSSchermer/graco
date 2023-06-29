@group(0) @binding(0)
var<uniform> fine_level_edge_ref_count: u32;

@group(0) @binding(1)
var<storage, read> validity_prefix_sum: array<u32>;

@group(0) @binding(2)
var<storage, read_write> coarse_level_edge_ref_count: u32;

@compute @workgroup_size(1, 1, 1)
fn main() {
    coarse_level_edge_ref_count = validity_prefix_sum[fine_level_edge_ref_count - 1];
}
