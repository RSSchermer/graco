use std::convert::TryInto;
use std::error::Error;

use arwa::dom::{selector, ParentNode};
use arwa::html::{HtmlCanvasElement, HtmlInputElement};
use arwa::window::window;
use empa::arwa::{
    AlphaMode, CanvasConfiguration, HtmlCanvasElementExt, NavigatorExt, RequestAdapterOptions,
};
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa::texture::format::rgba8unorm;
use empa::{abi, buffer, texture};
use futures::{FutureExt, StreamExt};
use graco::matching::{
    MatchPairsByEdgeWeight, MatchPairsByEdgeWeightConfig, MatchPairsByEdgeWeightInput,
    MatchPairsByEdgeWeightsCounts,
};
use web_viewer::{GraphRenderer, GraphRendererInput};
use std::str::FromStr;
use arwa::ui::UiEventTarget;

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
        nodes_position,
        nodes_edges,
        nodes_edge_weights,
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

    let GraphState {
        nodes_edge_offset,
        nodes_position,
        nodes_edges,
        nodes_edge_weights,
    } = generate_regular_graph_state(50, 0.45);

    let renderer = GraphRenderer::init(device.clone());

    let nodes_edge_offset: Buffer<[u32], _> =
        device.create_buffer(nodes_edge_offset, buffer::Usages::storage_binding());
    let nodes_edges: Buffer<[u32], _> =
        device.create_buffer(nodes_edges, buffer::Usages::storage_binding());
    let nodes_edge_weights: Buffer<[u32], _> = device.create_buffer(
        nodes_edge_weights,
        buffer::Usages::storage_binding().and_copy_src(),
    );
    let nodes_matching = device.create_slice_buffer_zeroed(
        nodes_edge_offset.len(),
        buffer::Usages::storage_binding().and_copy_src(),
    );
    let nodes_position: Buffer<[abi::Vec2<f32>], _> =
        device.create_buffer(nodes_position, buffer::Usages::storage_binding());

    let node_count = device.create_buffer(
        nodes_edge_offset.len() as u32,
        buffer::Usages::uniform_binding(),
    );
    let edge_ref_count =
        device.create_buffer(nodes_edges.len() as u32, buffer::Usages::uniform_binding());

    let rounds_input: HtmlInputElement = window.document().query_selector(&selector!("#rounds")).ok_or("rounds input not found")?.try_into()?;

    let match_and_render = || {
        let rounds = usize::from_str(&rounds_input.value()).unwrap();

        let mut matcher = MatchPairsByEdgeWeight::init(
            device.clone(),
            MatchPairsByEdgeWeightConfig {
                rounds,
                prng_seed: 1,
            },
        );

        let mut encoder = device.create_command_encoder();

        encoder = matcher.encode(
            encoder,
            MatchPairsByEdgeWeightInput {
                nodes_edge_offset: nodes_edge_offset.view(),
                nodes_edges: nodes_edges.view(),
                nodes_edge_weights: nodes_edge_weights.view(),
                count: Some(MatchPairsByEdgeWeightsCounts {
                    node_count: node_count.uniform(),
                    edge_ref_count: edge_ref_count.uniform(),
                }),
            },
            nodes_matching.view(),
        );

        encoder = renderer.encode(
            encoder,
            GraphRendererInput {
                output_texture: &context.get_current_texture(),
                node_count: node_count.view(),
                edge_ref_count: edge_ref_count.view(),
                nodes_edge_offset: nodes_edge_offset.view(),
                nodes_edges: nodes_edges.view(),
                nodes_matching: nodes_matching.view(),
                nodes_position: nodes_position.view(),
            },
        );

        device.queue().submit(encoder.finish());
    };

    match_and_render();

    let match_button = window.document().query_selector(&selector!("#match_button")).ok_or("match button not found")?;
    let mut match_clicks = match_button.on_click();

    while let Some(_) = match_clicks.next().await {
        match_and_render();
    }

    Ok(())
}
