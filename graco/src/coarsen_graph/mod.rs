mod collect_coarse_nodes_edge_weights;
mod compact_coarse_edges;
mod finalize_coarse_nodes_edge_offset;
mod gather_edge_owner_list;
mod generate_dispatches;
mod generate_index_list;
mod mark_coarse_edge_validity;
mod resolve_coarse_edge_ref_count;

mod coarsen_graph;
pub use self::coarsen_graph::{CoarsenCounts, CoarsenGraph, CoarsenGraphInput, CoarsenGraphOutput};

const DEFAULT_GROUP_SIZE: u32 = 256;
