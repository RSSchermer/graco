use empa::buffer::{ReadOnlyStorage, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};
use empa::{abi, buffer};

const GROUPS_SIZE: u32 = 256;

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources {
    #[resource(binding = 0, visibility = "COMPUTE")]
    node_count: Uniform<u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    edge_ref_count: Uniform<u32>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    nodes_edge_offset: ReadOnlyStorage<[u32]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    nodes_edges: ReadOnlyStorage<[u32]>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    nodes_position: ReadOnlyStorage<[abi::Vec2<f32>]>,
    #[resource(binding = 5, visibility = "COMPUTE")]
    nodes_edge_weights: Storage<[u32]>,
}

type ResourcesLayout = <Resources as empa::resource_binding::Resources>::Layout;

pub struct ComputeEdgeWeightsInput<'a, U0, U1, U2, U3, U4, U5> {
    pub node_count: buffer::View<'a, u32, U0>,
    pub edge_ref_count: buffer::View<'a, u32, U1>,
    pub nodes_edge_offset: buffer::View<'a, [u32], U2>,
    pub nodes_edges: buffer::View<'a, [u32], U3>,
    pub nodes_position: buffer::View<'a, [abi::Vec2<f32>], U4>,
    pub nodes_edge_weights: buffer::View<'a, [u32], U5>,
}

pub struct ComputeEdgeWeights {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl ComputeEdgeWeights {
    pub fn init(device: Device) -> Self {
        let shader = device.create_shader_module(&SHADER);

        let bind_group_layout = device.create_bind_group_layout::<ResourcesLayout>();
        let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

        let pipeline = device.create_compute_pipeline(
            &ComputePipelineDescriptorBuilder::begin()
                .layout(&pipeline_layout)
                .compute(&ComputeStageBuilder::begin(&shader, "main").finish())
                .finish(),
        );

        ComputeEdgeWeights {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4, U5>(
        &self,
        encoder: CommandEncoder,
        input: ComputeEdgeWeightsInput<U0, U1, U2, U3, U4, U5>,
    ) -> CommandEncoder
    where
        U0: buffer::UniformBinding,
        U1: buffer::UniformBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding,
        U5: buffer::StorageBinding,
    {
        let ComputeEdgeWeightsInput {
            node_count, edge_ref_count, nodes_edge_offset, nodes_edges, nodes_position, nodes_edge_weights
        } = input;

        let workgroups = (nodes_edge_offset.len() as u32).div_ceil(GROUPS_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                node_count: node_count.uniform(),
                edge_ref_count: edge_ref_count.uniform(),
                nodes_edge_offset: nodes_edge_offset.read_only_storage(),
                nodes_edges: nodes_edges.read_only_storage(),
                nodes_position: nodes_position.read_only_storage(),
                nodes_edge_weights: nodes_edge_weights.storage()
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
