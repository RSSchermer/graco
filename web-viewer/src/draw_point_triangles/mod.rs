use std::ops::Rem;

use empa::buffer::{Buffer, ReadOnlyStorage, Uniform};
use empa::command::{
    DrawIndexed, DrawIndexedCommandEncoder, RenderBundle, RenderBundleEncoderDescriptor,
    RenderStateEncoder, ResourceBindingCommandEncoder,
};
use empa::device::Device;
use empa::render_pipeline::{
    ColorOutput, ColorWriteMask, FragmentStageBuilder, IndexAny, RenderPipeline,
    RenderPipelineDescriptorBuilder, Vertex, VertexStageBuilder,
};
use empa::render_target::RenderLayout;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::texture::format::rgba8unorm;
use empa::type_flag::{O, X};
use empa::{abi, buffer};

use crate::{
    INDICES_PER_POINT, POINT_SIZE, RADIANS_PER_TRIANGLE, TRIANGLES_PER_POINT, VERTICES_PER_POINT,
};

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(Vertex, Clone, Copy)]
pub struct TriangleVertex {
    #[vertex_attribute(location = 0, format = "float32x2")]
    position: [f32; 2],
}

#[derive(empa::resource_binding::Resources)]
struct Resources {
    #[resource(binding = 0, visibility = "VERTEX")]
    transform: Uniform<abi::Mat3x3>,
    #[resource(binding = 1, visibility = "VERTEX")]
    nodes_position: ReadOnlyStorage<[abi::Vec2<f32>]>,
}

type ResourcesLayout = <Resources as empa::resource_binding::Resources>::Layout;

pub struct DrawPointTriangles {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline:
        RenderPipeline<RenderLayout<rgba8unorm, ()>, TriangleVertex, IndexAny, (ResourcesLayout,)>,
    triangle_vertices: Buffer<[TriangleVertex], buffer::Usages<O, O, O, O, X, O, O, O, O, O>>,
    triangle_indices: Buffer<[u16], buffer::Usages<O, O, O, O, O, X, O, O, O, O>>,
}

impl DrawPointTriangles {
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
                            .vertex_layout::<TriangleVertex>()
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
                    .finish(),
            )
            .await;

        let mut triangle_vertices = Vec::with_capacity(VERTICES_PER_POINT);

        triangle_vertices.push(TriangleVertex {
            position: [0.0, 0.0],
        });

        for i in 0..TRIANGLES_PER_POINT {
            let theta = i as f32 * RADIANS_PER_TRIANGLE;

            triangle_vertices.push(TriangleVertex {
                position: [POINT_SIZE * f32::sin(theta), POINT_SIZE * f32::cos(theta)],
            })
        }

        let triangle_vertices = device.create_buffer(triangle_vertices, buffer::Usages::vertex());

        let mut triangle_indices = Vec::with_capacity(INDICES_PER_POINT);

        for i in 0..TRIANGLES_PER_POINT as u16 {
            let current_index = i + 1;
            let next_index = (i + 1).rem(TRIANGLES_PER_POINT as u16) + 1;

            triangle_indices.push(0);
            triangle_indices.push(current_index);
            triangle_indices.push(next_index);
        }

        let triangle_indices = device.create_buffer(triangle_indices, buffer::Usages::index());

        DrawPointTriangles {
            device,
            bind_group_layout,
            pipeline,
            triangle_vertices,
            triangle_indices,
        }
    }

    pub fn render_bundle<U0, U1, U2>(
        &self,
        nodes_position: buffer::View<[abi::Vec2<f32>], U0>,
        transform: buffer::View<abi::Mat3x3, U1>,
        dispatch: buffer::View<DrawIndexed, U2>,
    ) -> RenderBundle<RenderLayout<rgba8unorm, ()>>
    where
        U0: buffer::StorageBinding,
        U1: buffer::UniformBinding,
        U2: buffer::Indirect,
    {
        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                transform: transform.uniform(),
                nodes_position: nodes_position.read_only_storage(),
            },
        );

        self.device
            .create_render_bundle_encoder(&RenderBundleEncoderDescriptor::new::<rgba8unorm>())
            .set_pipeline(&self.pipeline)
            .set_vertex_buffers(self.triangle_vertices.view())
            .set_index_buffer(self.triangle_indices.view())
            .set_bind_groups(&bind_group)
            .draw_indexed_indirect(dispatch)
            .finish()
    }
}
