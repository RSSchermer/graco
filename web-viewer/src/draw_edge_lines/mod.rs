use empa::buffer::Uniform;
use empa::command::{
    Draw, DrawCommandEncoder, RenderBundle, RenderBundleEncoderDescriptor, RenderStateEncoder,
    ResourceBindingCommandEncoder,
};
use empa::device::Device;
use empa::render_pipeline::{
    ColorOutput, ColorWriteMask, FragmentStageBuilder, IndexAny, PrimitiveAssembly, RenderPipeline,
    RenderPipelineDescriptorBuilder, VertexStageBuilder,
};
use empa::render_target::RenderLayout;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::texture::format::rgba8unorm;
use empa::{abi, buffer};

use crate::edge_vertex::EdgeVertex;

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources {
    #[resource(binding = 0, visibility = "VERTEX")]
    transform: Uniform<abi::Mat3x3>,
}

type ResourcesLayout = <Resources as empa::resource_binding::Resources>::Layout;

pub struct DrawEdgeLines {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline:
        RenderPipeline<RenderLayout<rgba8unorm, ()>, EdgeVertex, IndexAny, (ResourcesLayout,)>,
}

impl DrawEdgeLines {
    pub async fn init(device: Device) -> Self {
        let shader = device.create_shader_module(&SHADER);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = device
            .create_render_pipeline(
                &RenderPipelineDescriptorBuilder::begin()
                    .layout(&pipeline_layout)
                    .vertex(
                        &VertexStageBuilder::begin(&shader, "vert_main")
                            .vertex_layout::<EdgeVertex>()
                            .finish(),
                    )
                    .fragment(
                        &FragmentStageBuilder::begin(&shader, "frag_main")
                            .color_outputs(ColorOutput {
                                format: rgba8unorm,
                                write_mask: ColorWriteMask::ALL,
                            })
                            .finish(),
                    )
                    .primitive_assembly(PrimitiveAssembly::line_list())
                    .finish(),
            )
            .await;

        DrawEdgeLines {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn render_bundle<U0, U1, U2>(
        &self,
        edge_vertices: buffer::View<[EdgeVertex], U0>,
        transform: buffer::View<abi::Mat3x3, U1>,
        dispatch: buffer::View<Draw, U2>,
    ) -> RenderBundle<RenderLayout<rgba8unorm, ()>>
    where
        U0: buffer::Vertex,
        U1: buffer::UniformBinding,
        U2: buffer::Indirect,
    {
        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                transform: transform.uniform(),
            },
        );

        self.device
            .create_render_bundle_encoder(&RenderBundleEncoderDescriptor::new::<rgba8unorm>())
            .set_pipeline(&self.pipeline)
            .set_vertex_buffers(edge_vertices)
            .set_bind_groups(&bind_group)
            .draw_indirect(dispatch)
            .finish()
    }
}
