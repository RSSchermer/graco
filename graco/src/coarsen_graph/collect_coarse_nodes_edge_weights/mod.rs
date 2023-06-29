use empa::buffer;
use empa::buffer::{ReadOnlyStorage, Storage, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{
    ComputePipeline, ComputePipelineDescriptorBuilder, ComputeStageBuilder,
};
use empa::device::Device;
use empa::resource_binding::BindGroupLayout;
use empa::shader_module::{shader_source, ShaderSource};

use crate::coarsen_graph::DEFAULT_GROUP_SIZE;

const SHADER: ShaderSource = shader_source!("shader.wgsl");

#[derive(empa::resource_binding::Resources)]
pub struct CollectCoarseNodesEdgeWeightsResources {
    #[resource(binding = 0, visibility = "COMPUTE")]
    pub count: Uniform<u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    pub mapped_edges: ReadOnlyStorage<[u32]>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    pub mapped_edge_weights: ReadOnlyStorage<[u32]>,
    #[resource(binding = 3, visibility = "COMPUTE")]
    pub validity_prefix_sum: ReadOnlyStorage<[u32]>,
    #[resource(binding = 4, visibility = "COMPUTE")]
    pub coarse_nodes_edge_weights: Storage<[u32]>,
}

type ResourcesLayout =
    <CollectCoarseNodesEdgeWeightsResources as empa::resource_binding::Resources>::Layout;

pub struct CollectCoarseNodesEdgeWeights {
    device: Device,
    bind_group_layout: BindGroupLayout<ResourcesLayout>,
    pipeline: ComputePipeline<(ResourcesLayout,)>,
}

impl CollectCoarseNodesEdgeWeights {
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

        CollectCoarseNodesEdgeWeights {
            device,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn encode<U>(
        &self,
        encoder: CommandEncoder,
        resources: CollectCoarseNodesEdgeWeightsResources,
        dispatch_indirect: bool,
        dispatch: buffer::View<DispatchWorkgroups, U>,
        fallback_count: u32,
    ) -> CommandEncoder
    where
        U: buffer::Indirect,
    {
        let bind_group = self
            .device
            .create_bind_group(&self.bind_group_layout, resources);

        let encoder = encoder
            .begin_compute_pass()
            .set_pipeline(&self.pipeline)
            .set_bind_groups(&bind_group);

        if dispatch_indirect {
            encoder.dispatch_workgroups_indirect(dispatch).end()
        } else {
            encoder
                .dispatch_workgroups(DispatchWorkgroups {
                    count_x: fallback_count.div_ceil(DEFAULT_GROUP_SIZE),
                    count_y: 1,
                    count_z: 1,
                })
                .end()
        }
    }
}
