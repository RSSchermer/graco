#![feature(int_roundings)]

mod draw_edge_lines;
mod draw_point_triangles;
mod edge_vertex;
mod generate_dispatches;
mod generate_edge_line_vertices;

mod renderer;
pub use self::renderer::{GraphRenderer, GraphRendererInput};

const TRIANGLES_PER_POINT: usize = 12;
const POINT_SIZE: f32 = 0.0025;

const RADIANS_PER_TRIANGLE: f32 = 2.0 * std::f32::consts::PI / TRIANGLES_PER_POINT as f32;
const VERTICES_PER_POINT: usize = TRIANGLES_PER_POINT + 1;
const INDICES_PER_POINT: usize = TRIANGLES_PER_POINT * 3;
