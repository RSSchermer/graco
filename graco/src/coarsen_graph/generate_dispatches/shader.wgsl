struct DispatchWorkgroups {
    x: u32,
    y: u32,
    z: u32
}

@group(0) @binding(0)
var<uniform> group_size: u32;

@group(0) @binding(1)
var<uniform> node_count: u32;

@group(0) @binding(2)
var<uniform> edge_ref_count: u32;

@group(0) @binding(3)
var<storage, read_write> node_count_dispatch: DispatchWorkgroups;

@group(0) @binding(4)
var<storage, read_write> edge_ref_count_dispatch: DispatchWorkgroups;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let node_count_workgroups = div_ceil(node_count, group_size);

    node_count_dispatch = DispatchWorkgroups(node_count_workgroups, 1, 1);

    let edge_ref_count_workgroups = div_ceil(edge_ref_count, group_size);

    edge_ref_count_dispatch = DispatchWorkgroups(edge_ref_count_workgroups, 1, 1);
}
