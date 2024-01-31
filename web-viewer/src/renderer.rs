use std::future::join;

use empa::buffer::Buffer;
use empa::command::{CommandEncoder, Draw, DrawIndexed, RenderPassDescriptor};
use empa::device::Device;
use empa::render_target::{FloatAttachment, LoadOp, RenderTarget, StoreOp};
use empa::texture::format::rgba8unorm;
use empa::texture::{AttachableImageDescriptor, Texture2D};
use empa::type_flag::{O, X};
use empa::{abi, buffer, texture};
use empa_glam::ToAbi;
use glam::{Mat3, Vec2};

use crate::draw_edge_lines::DrawEdgeLines;
use crate::draw_point_triangles::DrawPointTriangles;
use crate::generate_dispatches::{GenerateDispatches, GenerateDispatchesResources};
use crate::generate_edge_line_vertices::{GenerateEdgeLineVertices, GenerateEdgeLineVerticesInput};
use crate::INDICES_PER_POINT;

pub struct GraphRendererInput<'a, U0, U1, U2, U3, U4, U5, U6> {
    pub output_texture: &'a Texture2D<rgba8unorm, U0>,
    pub node_count: buffer::View<'a, u32, U1>,
    pub edge_ref_count: buffer::View<'a, u32, U2>,
    pub nodes_edge_offset: buffer::View<'a, [u32], U3>,
    pub nodes_edges: buffer::View<'a, [u32], U4>,
    pub nodes_matching: buffer::View<'a, [u32], U5>,
    pub nodes_position: buffer::View<'a, [abi::Vec2<f32>], U6>,
}

pub struct GraphRenderer {
    device: Device,
    generate_dispatches: GenerateDispatches,
    generate_edge_line_vertices: GenerateEdgeLineVertices,
    draw_point_triangles: DrawPointTriangles,
    draw_edge_lines: DrawEdgeLines,
    triangle_index_count: Buffer<u32, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>,
    draw_point_triangles_dispatch:
        Buffer<DrawIndexed, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
    draw_edge_lines_dispatch: Buffer<Draw, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
}

impl GraphRenderer {
    pub async fn init(device: Device) -> Self {
        let (
            generate_dispatches,
            generate_edge_line_vertices,
            draw_point_triangles,
            draw_edge_lines,
        ) = join!(
            GenerateDispatches::init(device.clone()),
            GenerateEdgeLineVertices::init(device.clone()),
            DrawPointTriangles::init(device.clone()),
            DrawEdgeLines::init(device.clone()),
        )
        .await;

        let triangle_index_count =
            device.create_buffer(INDICES_PER_POINT as u32, buffer::Usages::uniform_binding());
        let draw_point_triangles_dispatch = device.create_buffer(
            DrawIndexed {
                index_count: 0,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );
        let draw_edge_lines_dispatch = device.create_buffer(
            Draw {
                vertex_count: 0,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );

        GraphRenderer {
            device,
            generate_dispatches,
            generate_edge_line_vertices,
            draw_point_triangles,
            draw_edge_lines,
            triangle_index_count,
            draw_point_triangles_dispatch,
            draw_edge_lines_dispatch,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4, U5, U6>(
        &self,
        mut encoder: CommandEncoder,
        input: GraphRendererInput<U0, U1, U2, U3, U4, U5, U6>,
    ) -> CommandEncoder
    where
        U0: texture::RenderAttachment,
        U1: buffer::UniformBinding,
        U2: buffer::UniformBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding,
        U5: buffer::StorageBinding,
        U6: buffer::StorageBinding,
    {
        let GraphRendererInput {
            output_texture,
            node_count,
            edge_ref_count,
            nodes_edge_offset,
            nodes_edges,
            nodes_matching,
            nodes_position,
        } = input;

        let edge_vertices = self.device.create_slice_buffer_zeroed(
            nodes_edges.len() * 2,
            buffer::Usages::storage_binding().and_vertex(),
        );

        encoder = self.generate_dispatches.encode(
            encoder,
            GenerateDispatchesResources {
                triangle_index_count: self.triangle_index_count.uniform(),
                node_count: node_count.uniform(),
                edge_ref_count: edge_ref_count.uniform(),
                draw_point_triangles_dispatch: self.draw_point_triangles_dispatch.storage(),
                draw_edge_lines_dispatch: self.draw_edge_lines_dispatch.storage(),
            },
        );

        encoder = self.generate_edge_line_vertices.encode(
            encoder,
            GenerateEdgeLineVerticesInput {
                node_count,
                edge_ref_count,
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

        let points_bundle = self.draw_point_triangles.render_bundle(
            nodes_position,
            transform.view(),
            self.draw_point_triangles_dispatch.view(),
        );
        let edges_bundle = self.draw_edge_lines.render_bundle(
            edge_vertices.view(),
            transform.view(),
            self.draw_edge_lines_dispatch.view(),
        );

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
