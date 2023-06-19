use empa::buffer;
use empa::buffer::{ReadOnlyStorage, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};

use crate::matching::match_pairs_by_edge_weight::match_state::MatchState;

const GROUPS_SIZE: u32 = 256;

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources {
    #[resource(binding = 0, visibility = "COMPUTE")]
    has_live_nodes: Uniform<u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    nodes_match_state: Storage<[MatchState]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    nodes_edge_offset: ReadOnlyStorage<[u32]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    nodes_edges: ReadOnlyStorage<[u32]>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    nodes_edge_weights: ReadOnlyStorage<[f32]>,
    #[resource(binding = 5, visibility = "COMPUTE")]
    nodes_proposal: Storage<[u32]>,
}

type ResourcesLayout = <Resources as empa::resource_binding::Resources>::Layout;

pub struct MakeProposalsInput<'a, U0, U1, U2, U3, U4, U5> {
    pub has_live_nodes: buffer::View<'a, u32, U0>,
    pub nodes_match_state: buffer::View<'a, [MatchState], U1>,
    pub nodes_edge_offset: buffer::View<'a, [u32], U2>,
    pub nodes_edges: buffer::View<'a, [u32], U3>,
    pub nodes_edge_weights: buffer::View<'a, [f32], U4>,
    pub nodes_proposal: buffer::View<'a, [u32], U5>,
}

pub struct MakeProposals {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl MakeProposals {
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

        MakeProposals {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4, U5>(
        &self,
        encoder: CommandEncoder,
        input: MakeProposalsInput<U0, U1, U2, U3, U4, U5>,
    ) -> CommandEncoder
    where
        U0: buffer::UniformBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding,
        U5: buffer::StorageBinding,
    {
        let MakeProposalsInput {
            has_live_nodes,
            nodes_match_state,
            nodes_edge_offset,
            nodes_edges,
            nodes_edge_weights,
            nodes_proposal,
        } = input;

        let workgroups = (nodes_match_state.len() as u32).div_ceil(GROUPS_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                has_live_nodes: has_live_nodes.uniform(),
                nodes_match_state: nodes_match_state.storage(),
                nodes_edge_offset: nodes_edge_offset.read_only_storage(),
                nodes_edges: nodes_edges.read_only_storage(),
                nodes_edge_weights: nodes_edge_weights.read_only_storage(),
                nodes_proposal: nodes_proposal.storage(),
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
