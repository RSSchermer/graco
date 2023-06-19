use empa::buffer;
use empa::buffer::Storage;
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};

const GROUPS_SIZE: u32 = 256;

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
struct Resources {
    #[resource(binding = 0, visibility = "COMPUTE")]
    nodes_match_state: Storage<[u32]>,
}

type ResourcesLayout = <Resources as empa::resource_binding::Resources>::Layout;

pub struct FinalizeMatching {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl FinalizeMatching {
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

        FinalizeMatching {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U>(
        &self,
        encoder: CommandEncoder,
        input: buffer::View<[u32], U>,
    ) -> CommandEncoder
    where
        U: buffer::StorageBinding,
    {
        let workgroups = (input.len() as u32).div_ceil(GROUPS_SIZE);

        let bind_group = self.device.create_bind_group(
            &self.bind_group_layout,
            Resources {
                nodes_match_state: input.storage(),
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
