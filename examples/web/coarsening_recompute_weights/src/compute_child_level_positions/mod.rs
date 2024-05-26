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

const GROUPS_SIZE: u32 = 256;

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources<'a> {
    #[resource(binding = 0, visibility = "COMPUTE")]
    child_level_node_count: Uniform<'a, u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    parent_level_node_count: Uniform<'a, u32>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    parent_level_positions: Storage<'a, [abi::Vec2<f32>]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    coarse_nodes_mapping_offset: Storage<'a, [u32]>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    coarse_nodes_mapping: Storage<'a, [u32]>,
    #[resource(binding = 5, visibility = "COMPUTE")]
    child_level_positions: Storage<'a, [abi::Vec2<f32>], ReadWrite>,
}

type ResourcesLayout = <Resources<'static> as empa::resource_binding::Resources>::Layout;

pub struct ComputeChildLevelPositionsInput<'a, U0, U1, U2, U3, U4, U5> {
    pub child_level_node_count: buffer::View<'a, u32, U0>,
    pub parent_level_node_count: buffer::View<'a, u32, U1>,
    pub parent_level_positions: buffer::View<'a, [abi::Vec2<f32>], U2>,
    pub coarse_nodes_mapping_offset: buffer::View<'a, [u32], U3>,
    pub coarse_nodes_mapping: buffer::View<'a, [u32], U4>,
    pub child_level_positions: buffer::View<'a, [abi::Vec2<f32>], U5>,
}

pub struct ComputeChildLevelPositions {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl ComputeChildLevelPositions {
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

        ComputeChildLevelPositions {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4, U5>(
        &self,
        encoder: CommandEncoder,
        input: ComputeChildLevelPositionsInput<U0, U1, U2, U3, U4, U5>,
    ) -> CommandEncoder
    where
        U0: buffer::UniformBinding,
        U1: buffer::UniformBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding,
        U5: buffer::StorageBinding,
    {
        let ComputeChildLevelPositionsInput {
            child_level_node_count,
            parent_level_node_count,
            parent_level_positions,
            coarse_nodes_mapping_offset,
            coarse_nodes_mapping,
            child_level_positions,
        } = input;

        let workgroups = (coarse_nodes_mapping_offset.len() as u32).div_ceil(GROUPS_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                child_level_node_count: child_level_node_count.uniform(),
                parent_level_node_count: parent_level_node_count.uniform(),
                parent_level_positions: parent_level_positions.storage(),
                coarse_nodes_mapping_offset: coarse_nodes_mapping_offset.storage(),
                coarse_nodes_mapping: coarse_nodes_mapping.storage(),
                child_level_positions: child_level_positions.storage(),
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
