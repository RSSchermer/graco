#![feature(future_join, int_roundings)]

mod compute_child_level_positions;

use std::convert::TryInto;
use std::error::Error;
use std::future::join;
use std::mem;

use arwa::console;
use arwa::dom::{selector, ParentNode};
use arwa::html::HtmlCanvasElement;
use arwa::ui::UiEventTarget;
use arwa::window::window;
use empa::arwa::{
    AlphaMode, CanvasConfiguration, HtmlCanvasElementExt, NavigatorExt, RequestAdapterOptions,
};
use empa::buffer::Buffer;
use empa::device::{Device, DeviceDescriptor};
use empa::texture::format::rgba8unorm;
use empa::type_flag::{O, X};
use empa::{abi, buffer, texture};
use futures::{FutureExt, StreamExt};
use graco::matching::{
    MatchPairsByEdgeWeight, MatchPairsByEdgeWeightInput, MatchPairsByEdgeWeightsCounts,
};
use graco::{CoarsenCounts, CoarsenGraph, CoarsenGraphInput, CoarsenGraphOutput};
use web_viewer::{GraphRenderer, GraphRendererInput};

use crate::compute_child_level_positions::{
    ComputeChildLevelPositions, ComputeChildLevelPositionsInput,
};

struct GraphState {
    nodes_edge_offset: Vec<u32>,
    nodes_position: Vec<abi::Vec2<f32>>,
    nodes_edges: Vec<u32>,
    nodes_edge_weights: Vec<u32>,
}

fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    arwa::spawn_local(render().map(|res| res.unwrap()));
}

struct GraphLevel {
    nodes_edge_offset: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, O, X, O, O>>,
    nodes_position: Buffer<[abi::Vec2<f32>], buffer::Usages<O, O, X, O, O, O, O, X, O, O>>,
    nodes_edges: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, O, X, O, O>>,
    nodes_edge_weights: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, X, X, O, O>>,
    node_count: Buffer<u32, buffer::Usages<O, O, X, X, O, O, O, X, O, O>>,
    edge_ref_count: Buffer<u32, buffer::Usages<O, O, X, X, O, O, O, X, O, O>>,
}

impl GraphLevel {
    fn from_data(device: &Device, state: &GraphState) -> Self {
        let GraphState {
            nodes_edge_offset,
            nodes_position,
            nodes_edges,
            nodes_edge_weights,
        } = state;

        let nodes_edge_offset = device.create_buffer(
            nodes_edge_offset.as_slice(),
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_position = device.create_buffer(
            nodes_position.as_slice(),
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
            nodes_position,
            nodes_edges,
            nodes_edge_weights,
            node_count,
            edge_ref_count,
        }
    }

    fn with_capacity(device: &Device, node_count: usize, edge_ref_count: usize) -> Self {
        let edge_capacity = usize::max(node_count, edge_ref_count);

        let nodes_edge_offset = device.create_slice_buffer_zeroed(
            node_count,
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_position = device.create_slice_buffer_zeroed(
            node_count,
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_edges = device.create_slice_buffer_zeroed(
            edge_capacity,
            buffer::Usages::storage_binding().and_copy_src(),
        );
        let nodes_edge_weights = device.create_slice_buffer_zeroed(
            edge_capacity,
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
            nodes_position,
            nodes_edges,
            nodes_edge_weights,
            node_count,
            edge_ref_count,
        }
    }
}

async fn render() -> Result<(), Box<dyn Error>> {
    let window = window();
    let empa = window.navigator().empa();
    let canvas: HtmlCanvasElement = window
        .document()
        .query_selector(&selector!("#canvas"))
        .ok_or("canvas not found")?
        .try_into()?;

    let adapter = empa
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .ok_or("adapter not found")?;
    let device = adapter.request_device(&DeviceDescriptor::default()).await?;

    let context = canvas.empa_context().configure(&CanvasConfiguration {
        device: &device,
        format: rgba8unorm,
        usage: texture::Usages::render_attachment(),
        view_formats: (),
        alpha_mode: AlphaMode::Opaque,
    });

    let (mut matcher, mut coarsen_graph, compute_child_level_positions, renderer) = join!(
        MatchPairsByEdgeWeight::init(device.clone(), Default::default()),
        CoarsenGraph::init(device.clone()),
        ComputeChildLevelPositions::init(device.clone()),
        GraphRenderer::init(device.clone()),
    )
    .await;

    let node_count = 3;
    let edge_ref_count = 0;

    let nodes_edge_offset = [0, 0, 0];

    let mut nodes_position = [
        abi::Vec2(-0.5, 0.0),
        abi::Vec2(0.0, 0.5),
        abi::Vec2(0.5, 0.0),
    ];

    for position in &mut nodes_position {
        position.0 = 0.5 * (position.0 + 1.0);
        position.1 = 0.5 * (position.1 + 1.0);
    }

    let base_level = GraphLevel {
        nodes_edge_offset: device.create_buffer(
            nodes_edge_offset.as_slice(),
            buffer::Usages::storage_binding().and_copy_src(),
        ),
        nodes_position: device.create_buffer(
            nodes_position.as_slice(),
            buffer::Usages::storage_binding().and_copy_src(),
        ),
        nodes_edges: device
            .create_slice_buffer_zeroed(3, buffer::Usages::storage_binding().and_copy_src()),
        nodes_edge_weights: device.create_slice_buffer_zeroed(
            3,
            buffer::Usages::storage_binding()
                .and_copy_src()
                .and_copy_dst(),
        ),
        node_count: device.create_buffer(
            node_count as u32,
            buffer::Usages::storage_binding()
                .and_uniform_binding()
                .and_copy_src(),
        ),
        edge_ref_count: device.create_buffer(
            0,
            buffer::Usages::storage_binding()
                .and_uniform_binding()
                .and_copy_src(),
        ),
    };

    let nodes_matching = device.create_slice_buffer_zeroed(
        node_count,
        buffer::Usages::storage_binding()
            .and_copy_src()
            .and_copy_dst(),
    );

    let mut encoder = device.create_command_encoder();

    encoder = matcher.encode(
        encoder,
        MatchPairsByEdgeWeightInput {
            nodes_edge_offset: base_level.nodes_edge_offset.view(),
            nodes_edges: base_level.nodes_edges.view(),
            nodes_edge_weights: base_level.nodes_edge_weights.view(),
            count: Some(MatchPairsByEdgeWeightsCounts {
                node_count: base_level.node_count.uniform(),
                edge_ref_count: base_level.edge_ref_count.uniform(),
            }),
        },
        nodes_matching.view(),
    );

    encoder = renderer.encode(
        encoder,
        GraphRendererInput {
            output_texture: &context.get_current_texture(),
            node_count: device
                .create_buffer(node_count as u32, buffer::Usages::uniform_binding())
                .view(),
            edge_ref_count: device
                .create_buffer(0, buffer::Usages::uniform_binding())
                .view(),
            nodes_edge_offset: base_level.nodes_edge_offset.view(),
            nodes_edges: base_level.nodes_edges.view(),
            nodes_matching: nodes_matching.view(),
            nodes_position: base_level.nodes_position.view(),
        },
    );

    let matching_readback =
        device.create_slice_buffer_zeroed(node_count, buffer::Usages::copy_dst().and_map_read());

    encoder = encoder.copy_buffer_to_buffer_slice(nodes_matching.view(), matching_readback.view());

    device.queue().submit(encoder.finish());

    let fine_nodes_mapping = device
        .create_slice_buffer_zeroed(node_count, buffer::Usages::storage_binding().and_copy_src());
    let coarse_nodes_mapping_offset = device
        .create_slice_buffer_zeroed(node_count, buffer::Usages::storage_binding().and_copy_src());
    let coarse_nodes_mapping = device
        .create_slice_buffer_zeroed(node_count, buffer::Usages::storage_binding().and_copy_src());

    let temporary_storage_0 = device.create_slice_buffer_zeroed(
        3,
        buffer::Usages::storage_binding()
            .and_copy_dst()
            .and_copy_src(),
    );
    let temporary_storage_1 = device.create_slice_buffer_zeroed(
        3,
        buffer::Usages::storage_binding()
            .and_copy_dst()
            .and_copy_src(),
    );

    let mut parent_level = base_level;
    let mut child_level = GraphLevel::with_capacity(&device, node_count, 1);

    let coarsen_button = window
        .document()
        .query_selector(&selector!("#coarsen_button"))
        .ok_or("coarsen button not found")?;

    let mut coarsen_clicks = coarsen_button.on_click();

    while let Some(_) = coarsen_clicks.next().await {
        matching_readback.map_read().await?;

        console::log!("Matching:", format!("{:#?}", &*matching_readback.mapped()));

        matching_readback.unmap();

        let mut encoder = device.create_command_encoder();

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

        encoder = compute_child_level_positions.encode(
            encoder,
            ComputeChildLevelPositionsInput {
                child_level_node_count: child_level.node_count.view(),
                parent_level_node_count: parent_level.node_count.view(),
                parent_level_positions: parent_level.nodes_position.view(),
                coarse_nodes_mapping_offset: coarse_nodes_mapping_offset.view(),
                coarse_nodes_mapping: coarse_nodes_mapping.view(),
                child_level_positions: child_level.nodes_position.view(),
            },
        );

        encoder = encoder.clear_buffer_slice(nodes_matching.view());

        encoder = matcher.encode(
            encoder,
            MatchPairsByEdgeWeightInput {
                nodes_edge_offset: child_level.nodes_edge_offset.view(),
                nodes_edges: child_level.nodes_edges.view(),
                nodes_edge_weights: child_level.nodes_edge_weights.view(),
                count: Some(MatchPairsByEdgeWeightsCounts {
                    node_count: child_level.node_count.uniform(),
                    edge_ref_count: child_level.edge_ref_count.uniform(),
                }),
            },
            nodes_matching.view(),
        );

        encoder =
            encoder.copy_buffer_to_buffer_slice(nodes_matching.view(), matching_readback.view());

        encoder = renderer.encode(
            encoder,
            GraphRendererInput {
                output_texture: &context.get_current_texture(),
                node_count: child_level.node_count.view(),
                edge_ref_count: child_level.edge_ref_count.view(),
                nodes_edge_offset: child_level.nodes_edge_offset.view(),
                nodes_edges: child_level.nodes_edges.view(),
                nodes_matching: nodes_matching.view(),
                nodes_position: child_level.nodes_position.view(),
            },
        );

        device.queue().submit(encoder.finish());

        mem::swap(&mut parent_level, &mut child_level);
    }

    Ok(())
}
