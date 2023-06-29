@group(0) @binding(0)
var<uniform> count: u32;

@group(0) @binding(1)
var<storage, read_write> data: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if index < count {
        data[index] = index;
    }
}
