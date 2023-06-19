use empa::abi;
use empa::render_pipeline::Vertex;
use zeroable::Zeroable;

#[derive(Vertex, abi::Sized, Clone, Copy, Zeroable, Debug)]
pub struct EdgeVertex {
    #[vertex_attribute(location = 0, format = "float32x2")]
    position: [f32; 2],
    #[vertex_attribute(location = 1, format = "float32x3")]
    color: [f32; 3],
}
