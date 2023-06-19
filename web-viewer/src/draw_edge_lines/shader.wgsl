struct VertexIn {
    @location(0) vertex_position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> transform: mat3x3<f32>;

@vertex
fn vert_main(input: VertexIn) -> VertexOut {
    let transformed = transform * vec3(input.vertex_position, 1.0);

    var result = VertexOut();

    result.position = vec4(transformed.xy, 0.0, 1.0);
    result.color = input.color;

    return result;
}

@fragment
fn frag_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4(color, 1.0);
}
