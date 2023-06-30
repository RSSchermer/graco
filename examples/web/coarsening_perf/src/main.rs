#![feature(int_roundings)]

use std::error::Error;

use arwa::window::window;
use empa::adapter::Features;
use empa::arwa::{NavigatorExt, PowerPreference, RequestAdapterOptions};
use empa::buffer::Buffer;
use empa::device::{Device, DeviceDescriptor};
use empa::type_flag::{O, X};
use empa::{abi, buffer};
use futures::FutureExt;
use graco::matching::{MatchPairsByEdgeWeight, MatchPairsByEdgeWeightInput};
use graco::{CoarsenCounts, CoarsenGraph, CoarsenGraphInput, CoarsenGraphOutput};

struct GraphState {
    nodes_edge_offset: Vec<u32>,
    nodes_edges: Vec<u32>,
    nodes_edge_weights: Vec<u32>,
}

fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    arwa::spawn_local(render().map(|res| res.unwrap()));
}

fn generate_regular_graph_state(grid_size: u32, perturbation_factor: f32) -> GraphState {
    let mut nodes_edge_offset = Vec::new();
    let mut nodes_position = Vec::new();
    let mut nodes_edges = Vec::new();
    let mut nodes_edge_weights = Vec::new();

    let mut rng = oorandom::Rand32::new(1);

    let spacing = 1.0 / (grid_size as f32 + 1.0);
    let max_perturbation = perturbation_factor * spacing;

    let compute_node_index = |col, row| -> u32 { row * grid_size + col };

    let mut compute_node_position = |col, row| -> abi::Vec2<f32> {
        let perturbation_x = rng.rand_float() * max_perturbation * 2.0 - max_perturbation;
        let perturbation_y = rng.rand_float() * max_perturbation * 2.0 - max_perturbation;

        abi::Vec2(
            spacing + col as f32 * spacing + perturbation_x,
            spacing + row as f32 * spacing + perturbation_y,
        )
    };

    fn compute_edge_weight(pos_a: abi::Vec2<f32>, pos_b: abi::Vec2<f32>) -> u32 {
        let d_x = pos_b.0 - pos_a.0;
        let d_y = pos_b.1 - pos_a.1;

        let distance = (d_x * d_x + d_y * d_y).sqrt();
        let distance_inv = 1.0 / distance;

        (distance_inv * 1000.0) as u32
    }

    for row in 0..grid_size {
        for col in 0..grid_size {
            nodes_position.push(compute_node_position(col, row));
        }
    }

    let mut current_edge_offset = 0;

    for row in 0..grid_size {
        for col in 0..grid_size {
            let current_index = compute_node_index(col, row);
            let current_pos = nodes_position[current_index as usize];

            nodes_edge_offset.push(current_edge_offset);

            let mut add_edge = |col, row| {
                let other_index = compute_node_index(col, row);
                let other_pos = nodes_position[other_index as usize];

                let weight = compute_edge_weight(current_pos, other_pos);

                nodes_edges.push(other_index);
                nodes_edge_weights.push(weight);

                current_edge_offset += 1;
            };

            if col > 0 {
                add_edge(col - 1, row);
            }

            if col < grid_size - 1 {
                add_edge(col + 1, row);
            }

            if row > 0 {
                add_edge(col, row - 1);
            }

            if row < grid_size - 1 {
                add_edge(col, row + 1);
            }
        }
    }

    GraphState {
        nodes_edge_offset,
        nodes_edges,
        nodes_edge_weights,
    }
}

struct GraphLevel {
    nodes_edge_offset: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, O, X, O, O>>,
    nodes_edges: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, O, X, O, O>>,
    nodes_edge_weights: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, X, X, O, O>>,
    node_count: Buffer<u32, buffer::Usages<O, O, X, X, O, O, O, X, O, O>>,
    edge_ref_count: Buffer<u32, buffer::Usages<O, O, X, X, O, O, O, X, O, O>>,
}

impl GraphLevel {
    fn from_data(device: &Device, state: &GraphState) -> Self {
        let GraphState {
            nodes_edge_offset,
            nodes_edges,
            nodes_edge_weights,
            ..
        } = state;

        let nodes_edge_offset = device.create_buffer(
            nodes_edge_offset.as_slice(),
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_edges = device.create_buffer(
            nodes_edges.as_slice(),
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_edge_weights = device.create_buffer(
            nodes_edge_weights.as_slice(),
            buffer::Usages::storage_binding()
                .and_copy_dst()
                .and_copy_src(),
        );
        let node_count = device.create_buffer(
            nodes_edge_offset.len() as u32,
            buffer::Usages::uniform_binding()
                .and_storage_binding()
                .and_copy_src(),
        );
        let edge_ref_count = device.create_buffer(
            nodes_edges.len() as u32,
            buffer::Usages::uniform_binding()
                .and_storage_binding()
                .and_copy_src(),
        );

        GraphLevel {
            nodes_edge_offset,
            nodes_edges,
            nodes_edge_weights,
            node_count,
            edge_ref_count,
        }
    }

    fn with_capacity(device: &Device, node_count: usize, edge_ref_count: usize) -> Self {
        let nodes_edge_offset = device.create_slice_buffer_zeroed(
            node_count,
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_edges = device.create_slice_buffer_zeroed(
            edge_ref_count,
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_edge_weights = device.create_slice_buffer_zeroed(
            edge_ref_count,
            buffer::Usages::storage_binding()
                .and_copy_dst()
                .and_copy_src(),
        );
        let node_count = device.create_buffer(
            0,
            buffer::Usages::uniform_binding()
                .and_storage_binding()
                .and_copy_src(),
        );
        let edge_ref_count = device.create_buffer(
            0,
            buffer::Usages::uniform_binding()
                .and_storage_binding()
                .and_copy_src(),
        );

        GraphLevel {
            nodes_edge_offset,
            nodes_edges,
            nodes_edge_weights,
            node_count,
            edge_ref_count,
        }
    }
}

async fn render() -> Result<(), Box<dyn Error>> {
    let grid_size = 1000;
    let node_count = grid_size * grid_size;

    arwa::console::log!("Matching a coarsening a graph with %i nodes...", node_count);

    let window = window();
    let empa = window.navigator().empa();

    let adapter = empa
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("adapter not found")?;
    let device = adapter
        .request_device(&DeviceDescriptor {
            required_features: Features::TIMESTAMP_QUERY,
            required_limits: Default::default(),
        })
        .await?;

    let mut matcher = MatchPairsByEdgeWeight::init(device.clone(), Default::default());
    let mut coarsen_graph = CoarsenGraph::init(device.clone());

    let graph_state = generate_regular_graph_state(grid_size, 0.45);
    let parent_level = GraphLevel::from_data(&device, &graph_state);

    let node_count = graph_state.nodes_edge_offset.len();
    let edge_ref_count = graph_state.nodes_edges.len();

    let child_level = GraphLevel::with_capacity(&device, node_count, edge_ref_count);

    let nodes_matching = device.create_slice_buffer_zeroed(
        node_count,
        buffer::Usages::storage_binding()
            .and_copy_src()
            .and_copy_dst(),
    );

    let fine_nodes_mapping = device
        .create_slice_buffer_zeroed(node_count, buffer::Usages::storage_binding().and_copy_src());
    let coarse_nodes_mapping_offset = device
        .create_slice_buffer_zeroed(node_count, buffer::Usages::storage_binding().and_copy_src());
    let coarse_nodes_mapping = device
        .create_slice_buffer_zeroed(node_count, buffer::Usages::storage_binding().and_copy_src());

    let temporary_storage_0 = device.create_slice_buffer_zeroed(
        edge_ref_count,
        buffer::Usages::storage_binding()
            .and_copy_dst()
            .and_copy_src(),
    );
    let temporary_storage_1 = device.create_slice_buffer_zeroed(
        edge_ref_count,
        buffer::Usages::storage_binding()
            .and_copy_dst()
            .and_copy_src(),
    );

    let timestamp_query_set = device.create_timestamp_query_set(3);
    let timestamps =
        device.create_slice_buffer_zeroed(3, buffer::Usages::query_resolve().and_copy_src());
    let timestamps_readback =
        device.create_slice_buffer_zeroed(3, buffer::Usages::copy_dst().and_map_read());

    let mut encoder = device.create_command_encoder();

    encoder = encoder.write_timestamp(&timestamp_query_set, 0);

    encoder = matcher.encode(
        encoder,
        MatchPairsByEdgeWeightInput {
            nodes_edge_offset: parent_level.nodes_edge_offset.view(),
            nodes_edges: parent_level.nodes_edges.view(),
            nodes_edge_weights: parent_level.nodes_edge_weights.view(),
            count: None,
        },
        nodes_matching.view(),
    );

    encoder = encoder.write_timestamp(&timestamp_query_set, 1);

    encoder = coarsen_graph.encode(
        encoder,
        CoarsenGraphInput {
            fine_nodes_edge_offset: parent_level.nodes_edge_offset.view(),
            fine_nodes_edges: parent_level.nodes_edges.view(),
            fine_nodes_edge_weights: parent_level.nodes_edge_weights.view(),
            fine_nodes_matching: nodes_matching.view(),
            temporary_storage_0: temporary_storage_0.view(),
            temporary_storage_1: temporary_storage_1.view(),
            counts: Some(CoarsenCounts {
                node_count: parent_level.node_count.uniform(),
                edge_ref_count: parent_level.edge_ref_count.uniform(),
            }),
        },
        CoarsenGraphOutput {
            fine_nodes_mapping: fine_nodes_mapping.view(),
            coarse_nodes_mapping_offset: coarse_nodes_mapping_offset.view(),
            coarse_nodes_mapping: coarse_nodes_mapping.view(),
            coarse_node_count: child_level.node_count.view(),
            coarse_edge_ref_count: child_level.edge_ref_count.view(),
            coarse_nodes_edge_offset: child_level.nodes_edge_offset.view(),
            coarse_nodes_edges: child_level.nodes_edges.view(),
            coarse_nodes_edge_weights: child_level.nodes_edge_weights.view(),
        },
    );

    encoder = encoder.write_timestamp(&timestamp_query_set, 2);

    encoder = encoder.resolve_timestamp_query_set(&timestamp_query_set, 0, timestamps.view());
    encoder = encoder.copy_buffer_to_buffer_slice(timestamps.view(), timestamps_readback.view());

    device.queue().submit(encoder.finish());

    timestamps_readback.map_read().await?;

    {
        let timestamps = timestamps_readback.mapped();
        let match_time = timestamps[1] - timestamps[0];
        let coarsening_time = timestamps[2] - timestamps[1];

        arwa::console::log!("Time to match: %i nanoseconds", match_time);
        arwa::console::log!("Time to coarsen: %i nanoseconds", coarsening_time);
    }

    timestamps_readback.unmap();

    Ok(())
}
