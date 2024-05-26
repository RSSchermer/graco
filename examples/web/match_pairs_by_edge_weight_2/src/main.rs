#![feature(async_closure)]

use std::convert::TryInto;
use std::error::Error;
use std::str::FromStr;

use arwa::dom::{selector, ParentNode};
use arwa::html::{HtmlCanvasElement, HtmlInputElement};
use arwa::ui::UiEventTarget;
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

fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    arwa::spawn_local(render().map(|res| res.unwrap()));
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

    let nodes_edge_offset = [
        0, 4, 8, 13, 18, 20, 24, 29, 34, 40, 46, 52, 57, 60, 64, 68, 74, 79, 83, 88, 93, 96, 100,
        43, 44, 46, 49, 50, 52, 54, 56, 60, 62, 62, 66, 69, 71, 74, 77, 139, 142, 145, 149, 153,
        157, 161, 165, 169, 172, 175, 179, 183, 187, 191, 195, 199, 202, 205, 209, 213, 217, 221,
        225, 229, 232, 234, 237, 240, 243, 246, 249, 252, 254,
    ];

    let nodes_edges = [
        1, 5, 10, 13, 0, 2, 5, 8, 1, 3, 6, 8, 14, 2, 4, 6, 7, 9, 3, 7, 0, 1, 8, 10, 2, 3, 9, 14,
        16, 3, 4, 9, 11, 12, 1, 2, 5, 10, 14, 15, 3, 6, 7, 11, 16, 18, 0, 5, 8, 13, 15, 19, 7, 9,
        12, 17, 18, 7, 11, 17, 0, 10, 19, 22, 2, 6, 8, 15, 8, 10, 14, 16, 19, 20, 6, 9, 15, 18, 21,
        11, 12, 18, 21, 9, 11, 16, 17, 21, 10, 13, 15, 20, 22, 15, 19, 21, 16, 17, 18, 20, 13, 19,
        1796, 3420, 1710, 1796, 1796, 3420, 1796, 1796, 1710, 1710, 1710, 1796, 1796, 1710, 1796,
        1796, 1710, 1710, 1796, 1710, 1796, 1710, 1796, 1710, 1710, 3420, 1710, 1796, 1796, 1710,
        1796, 1710, 3420, 1710, 1796, 1710, 1796, 1796, 1796, 1710, 1710, 1710, 1796, 1796, 1710,
        1710, 1710, 3420, 3420, 1710, 1796, 1796, 1796, 1710, 1710, 1796, 1710, 1710, 1710, 1710,
        1796, 1710, 1710, 3420, 1796, 1710, 1710, 1796, 1796, 3420, 1710, 1710, 1796, 1710, 41, 50,
        57, 49, 42, 51, 58, 50, 43, 52, 59, 51, 44, 53, 60, 52, 45, 54, 61, 53, 46, 55, 62, 54, 47,
        63, 48, 57, 64, 56, 49, 58, 65, 57, 50, 59, 66, 58, 51, 60, 67, 59, 52, 61, 68, 60, 53, 62,
        69, 61, 54, 63, 70, 62, 55, 71, 56, 65, 64, 57, 66, 65, 58, 67, 66, 59, 68, 67, 60, 69, 68,
        61, 70, 69, 62, 71, 70, 63,
    ];

    let nodes_edge_weights = [
        806, 1023, 606, 547, 806, 806, 1180, 674, 806, 806, 785, 690, 839, 806, 890, 790, 822, 674,
        890, 1059, 1023, 1180, 808, 717, 785, 790, 785, 949, 717, 822, 1059, 722, 790, 948, 674,
        690, 808, 785, 1099, 686, 674, 785, 722, 905, 808, 674, 606, 717, 785, 785, 720, 674, 790,
        905, 1120, 694, 790, 948, 1120, 717, 547, 785, 722, 839, 839, 949, 1099, 955, 686, 720,
        955, 937, 855, 969, 717, 808, 937, 972, 830, 694, 717, 848, 688, 674, 790, 972, 848, 974,
        674, 722, 855, 1033, 1120, 969, 1033, 837, 830, 688, 974, 837, 839, 1120, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    let mut nodes_position = [
        abi::Vec2(-0.875, -0.5555555),
        abi::Vec2(-0.49999997, -0.7777778),
        abi::Vec2(-0.125, -0.5555555),
        abi::Vec2(0.25, -0.77777785),
        abi::Vec2(0.75, -0.88888884),
        abi::Vec2(-0.62500006, -0.55555564),
        abi::Vec2(0.12499999, -0.33333337),
        abi::Vec2(0.75, -0.5555556),
        abi::Vec2(-0.37500003, -0.111111075),
        abi::Vec2(0.37500003, -0.111111075),
        abi::Vec2(-0.625, 0.11111113),
        abi::Vec2(0.625, 3.6572473e-8),
        abi::Vec2(0.875, -0.111111075),
        abi::Vec2(-0.875, 0.33333337),
        abi::Vec2(-0.125, 3.220864e-8),
        abi::Vec2(-0.24999999, 0.4444445),
        abi::Vec2(0.125, 0.33333337),
        abi::Vec2(0.875, 0.55555564),
        abi::Vec2(0.5, 0.5555556),
        abi::Vec2(-0.5, 0.77777785),
        abi::Vec2(-0.125, 0.77777785),
        abi::Vec2(0.37500006, 0.7777779),
        abi::Vec2(-0.875, 0.888889),
    ];

    for position in &mut nodes_position {
        position.0 = 0.5 * (position.0 + 1.0);
        position.1 = 0.5 * (position.1 + 1.0);
    }

    let renderer = GraphRenderer::init(device.clone()).await;

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

    let node_count = device.create_buffer(23, buffer::Usages::uniform_binding());
    let edge_ref_count = device.create_buffer(102, buffer::Usages::uniform_binding());

    let rounds_input: HtmlInputElement = window
        .document()
        .query_selector(&selector!("#rounds"))
        .ok_or("rounds input not found")?
        .try_into()?;

    let match_and_render = async || {
        let rounds = usize::from_str(&rounds_input.value()).unwrap();

        let mut matcher = MatchPairsByEdgeWeight::init(
            device.clone(),
            MatchPairsByEdgeWeightConfig {
                rounds,
                prng_seed: 1,
            },
        )
        .await;

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

    match_and_render().await;

    let match_button = window
        .document()
        .query_selector(&selector!("#match_button"))
        .ok_or("match button not found")?;
    let mut match_clicks = match_button.on_click();

    while let Some(_) = match_clicks.next().await {
        match_and_render().await;
    }

    Ok(())
}
