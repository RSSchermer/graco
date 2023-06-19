use empa::command::{CommandEncoder, RenderPassDescriptor};
use empa::device::Device;
use empa::render_target::{FloatAttachment, LoadOp, RenderTarget, StoreOp};
use empa::texture::format::rgba8unorm;
use empa::texture::{AttachableImageDescriptor, Texture2D};
use empa::{abi, buffer, texture};
use empa_glam::ToAbi;
use glam::{Mat3, Vec2};

use crate::draw_edge_lines::DrawEdgeLines;
use crate::draw_point_triangles::DrawPointTriangles;
use crate::generate_edge_line_vertices::{GenerateEdgeLineVertices, GenerateEdgeLineVerticesInput};

pub struct GraphRendererInput<'a, U0, U1, U2, U3, U4> {
    pub output_texture: &'a Texture2D<rgba8unorm, U0>,
    pub nodes_edge_offset: buffer::View<'a, [u32], U1>,
    pub nodes_edges: buffer::View<'a, [u32], U2>,
    pub nodes_matching: buffer::View<'a, [u32], U3>,
    pub nodes_position: buffer::View<'a, [abi::Vec2<f32>], U4>,
}

pub struct GraphRenderer {
    device: Device,
    generate_edge_line_vertices: GenerateEdgeLineVertices,
    draw_point_triangles: DrawPointTriangles,
    draw_edge_lines: DrawEdgeLines,
}

impl GraphRenderer {
    pub fn init(device: Device) -> Self {
        let generate_edge_line_vertices = GenerateEdgeLineVertices::init(device.clone());
        let draw_point_triangles = DrawPointTriangles::init(device.clone());
        let draw_edge_lines = DrawEdgeLines::init(device.clone());

        GraphRenderer {
            device,
            generate_edge_line_vertices,
            draw_point_triangles,
            draw_edge_lines,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4>(
        &self,
        mut encoder: CommandEncoder,
        input: GraphRendererInput<U0, U1, U2, U3, U4>,
    ) -> CommandEncoder
    where
        U0: texture::RenderAttachment,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding,
    {
        let GraphRendererInput {
            output_texture,
            nodes_edge_offset,
            nodes_edges,
            nodes_matching,
            nodes_position,
        } = input;

        let edge_vertices = self.device.create_slice_buffer_zeroed(
            nodes_edges.len() * 2,
            buffer::Usages::storage_binding().and_vertex(),
        );

        encoder = self.generate_edge_line_vertices.encode(
            encoder,
            GenerateEdgeLineVerticesInput {
                nodes_edge_offset,
                nodes_edges,
                nodes_position,
                nodes_matching,
                edge_vertices: edge_vertices.view(),
            },
        );

        let translation = Mat3::from_translation(Vec2::new(-1.0, -1.0));
        let scale = Mat3::from_scale(Vec2::new(2.0, 2.0));
        let transform = translation * scale;

        let transform = self
            .device
            .create_buffer(transform.to_abi(), buffer::Usages::uniform_binding());

        let points_bundle = self
            .draw_point_triangles
            .render_bundle(nodes_position, transform.view());
        let edges_bundle = self
            .draw_edge_lines
            .render_bundle(edge_vertices.view(), transform.view());

        encoder = encoder
            .begin_render_pass(&RenderPassDescriptor::new(&RenderTarget {
                color: FloatAttachment {
                    image: &output_texture.attachable_image(&AttachableImageDescriptor::default()),
                    load_op: LoadOp::Clear([1.0; 4]),
                    store_op: StoreOp::Store,
                },
                depth_stencil: (),
            }))
            .execute_bundle(&points_bundle)
            .execute_bundle(&edges_bundle)
            .end();

        encoder
    }
}
