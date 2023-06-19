use std::convert::TryInto;
use std::error::Error;

use arwa::dom::{selector, ParentNode};
use arwa::fetch::{FetchContext, Request};
use arwa::html::HtmlCanvasElement;
use arwa::url::Url;
use arwa::window::window;
use empa::arwa::{
    AlphaMode, CanvasConfiguration, HtmlCanvasElementExt, NavigatorExt, RequestAdapterOptions,
};
use empa::buffer::Buffer;
use empa::device::DeviceDescriptor;
use empa::texture::format::rgba8unorm;
use empa::{abi, buffer, texture};
use futures::FutureExt;
use graco::matching::{MatchPairsByEdgeWeight, MatchPairsByEdgeWeightInput};
use graph_loading::GraphData;
use web_viewer::{GraphRenderer, GraphRendererInput};

struct GraphState {
    nodes_edge_offset: Vec<u32>,
    nodes_position: Vec<abi::Vec2<f32>>,
    nodes_edges: Vec<u32>,
    nodes_edge_weights: Vec<f32>,
}

fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    arwa::spawn_local(render().map(|res| res.unwrap()));
}

async fn load_graph_state_from_file(filename: &str) -> Result<GraphState, Box<dyn Error>> {
    let url = Url::parse_with_base(filename, &window().location().to_url())?;
    let graph_response = window()
        .fetch(&Request::init(&url, Default::default()))
        .await?;
    let graph_src = graph_response.body().to_string().await?;
    let GraphData {
        nodes_edge_offset,
        nodes_edges,
    } = graph_loading::parse_graph_data(&graph_src).ok_or("invalid graph data")?;

    let node_count = nodes_edge_offset.len();

    let mut nodes_position = Vec::with_capacity(node_count);
    let mut rng = oorandom::Rand32::new(7);

    for _ in 0..node_count {
        nodes_position.push(abi::Vec2(rng.rand_float(), rng.rand_float()));
    }

    let mut nodes_edge_weights = vec![0.0; nodes_edges.len()];

    for i in 0..node_count {
        let edges_start = nodes_edge_offset[i];

        let edges_end = if i < node_count - 1 {
            nodes_edge_offset[i + 1]
        } else {
            nodes_edges.len() as u32
        };

        let pos_a = nodes_position[i];

        for j in edges_start..edges_end {
            let other_index = nodes_edges[j as usize];

            let pos_b = nodes_position[other_index as usize];

            let d_x = pos_b.0 - pos_a.0;
            let d_y = pos_b.1 - pos_a.1;

            let weight = (d_x * d_x + d_y * d_y).sqrt();

            nodes_edge_weights[j as usize] = 1.0 / weight;
        }
    }

    Ok(GraphState {
        nodes_edge_offset,
        nodes_position,
        nodes_edges,
        nodes_edge_weights,
    })
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

    fn compute_edge_weight(pos_a: abi::Vec2<f32>, pos_b: abi::Vec2<f32>) -> f32 {
        let d_x = pos_b.0 - pos_a.0;
        let d_y = pos_b.1 - pos_a.1;

        let distance = (d_x * d_x + d_y * d_y).sqrt();

        1.0 / distance
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

    // let GraphState {
    //     nodes_edge_offset, nodes_position, nodes_edges, nodes_edge_weights
    // } = load_graph_state_from_file("karate.graph").await?;

    let GraphState {
        nodes_edge_offset,
        nodes_position,
        nodes_edges,
        nodes_edge_weights,
    } = generate_regular_graph_state(50, 0.45);

    let mut matcher = MatchPairsByEdgeWeight::init(device.clone(), Default::default());
    let renderer = GraphRenderer::init(device.clone());

    let mut encoder = device.create_command_encoder();

    let nodes_edge_offset: Buffer<[u32], _> =
        device.create_buffer(nodes_edge_offset, buffer::Usages::storage_binding());
    let nodes_edges: Buffer<[u32], _> =
        device.create_buffer(nodes_edges, buffer::Usages::storage_binding());
    let nodes_edge_weights: Buffer<[f32], _> =
        device.create_buffer(nodes_edge_weights, buffer::Usages::storage_binding());
    let nodes_matching = device
        .create_slice_buffer_zeroed(nodes_edge_offset.len(), buffer::Usages::storage_binding());
    let nodes_position: Buffer<[abi::Vec2<f32>], _> =
        device.create_buffer(nodes_position, buffer::Usages::storage_binding());

    encoder = matcher.encode(
        encoder,
        MatchPairsByEdgeWeightInput {
            nodes_edge_offset: nodes_edge_offset.view(),
            nodes_edges: nodes_edges.view(),
            nodes_edge_weights: nodes_edge_weights.view(),
        },
        nodes_matching.view(),
    );

    encoder = renderer.encode(
        encoder,
        GraphRendererInput {
            output_texture: &context.get_current_texture(),
            nodes_edge_offset: nodes_edge_offset.view(),
            nodes_edges: nodes_edges.view(),
            nodes_matching: nodes_matching.view(),
            nodes_position: nodes_position.view(),
        },
    );

    device.queue().submit(encoder.finish());

    Ok(())
}
