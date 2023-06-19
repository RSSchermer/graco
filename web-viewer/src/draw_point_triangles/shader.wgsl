struct VertexIn {
    @location(0) vertex_position: vec2<f32>,
    @builtin(instance_index) instance_index: u32
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> transform: mat3x3<f32>;

@group(0) @binding(1)
var<storage, read> nodes_position: array<vec2<f32>>;

@vertex
fn vert_main(input: VertexIn) -> VertexOut {
    let translated = input.vertex_position + nodes_position[input.instance_index];
    let transformed = transform * vec3(translated, 1.0);

    var result = VertexOut();

    result.position = vec4(transformed.xy, 0.0, 1.0);

    return result;
}

@fragment
fn frag_main() -> @location(0) vec4<f32> {
    return vec4(0.0, 0.0, 0.0, 1.0);
}
