use empa::access_mode::ReadWrite;
use empa::buffer::{Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::{abi, buffer};

use crate::edge_vertex::EdgeVertex;

const GROUP_SIZE: u32 = 256;

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources<'a> {
    #[resource(binding = 0, visibility = "COMPUTE")]
    node_count: Uniform<'a, u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    edge_ref_count: Uniform<'a, u32>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    nodes_edge_offset: Storage<'a, [u32]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    nodes_edges: Storage<'a, [u32]>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    nodes_position: Storage<'a, [abi::Vec2<f32>]>,
    #[resource(binding = 5, visibility = "COMPUTE")]
    nodes_matching: Storage<'a, [u32]>,
    #[resource(binding = 6, visibility = "COMPUTE")]
    edge_vertices: Storage<'a, [EdgeVertex], ReadWrite>,
}

type ResourcesLayout = <Resources<'static> as empa::resource_binding::Resources>::Layout;

pub struct GenerateEdgeLineVerticesInput<'a, U0, U1, U2, U3, U4, U5, U6> {
    pub node_count: buffer::View<'a, u32, U0>,
    pub edge_ref_count: buffer::View<'a, u32, U1>,
    pub nodes_edge_offset: buffer::View<'a, [u32], U2>,
    pub nodes_edges: buffer::View<'a, [u32], U3>,
    pub nodes_position: buffer::View<'a, [abi::Vec2<f32>], U4>,
    pub nodes_matching: buffer::View<'a, [u32], U5>,
    pub edge_vertices: buffer::View<'a, [EdgeVertex], U6>,
}

pub struct GenerateEdgeLineVertices {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl GenerateEdgeLineVertices {
    pub async fn init(device: Device) -> Self {
        let shader = device.create_shader_module(&SHADER);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = device
            .create_compute_pipeline(
                &ComputePipelineDescriptorBuilder::begin()
                    .layout(&pipeline_layout)
                    .compute(ComputeStageBuilder::begin(&shader, "main").finish())
                    .finish(),
            )
            .await;

        GenerateEdgeLineVertices {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4, U5, U6>(
        &self,
        encoder: CommandEncoder,
        input: GenerateEdgeLineVerticesInput<U0, U1, U2, U3, U4, U5, U6>,
    ) -> CommandEncoder
    where
        U0: buffer::UniformBinding,
        U1: buffer::UniformBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding,
        U5: buffer::StorageBinding,
        U6: buffer::StorageBinding,
    {
        let GenerateEdgeLineVerticesInput {
            node_count,
            edge_ref_count,
            nodes_edge_offset,
            nodes_edges,
            nodes_position,
            nodes_matching,
            edge_vertices,
        } = input;

        let workgroups = (nodes_edge_offset.len() as u32).div_ceil(GROUP_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                node_count: node_count.uniform(),
                edge_ref_count: edge_ref_count.uniform(),
                nodes_edge_offset: nodes_edge_offset.storage(),
                nodes_edges: nodes_edges.storage(),
                nodes_position: nodes_position.storage(),
                nodes_matching: nodes_matching.storage(),
                edge_vertices: edge_vertices.storage(),
            },
        );

        encoder
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group)
            .dispatch_workgroups(DispatchWorkgroups {
                count_x: workgroups,
                count_y: 1,
                count_z: 1,
            })
            .end()
    }
}
