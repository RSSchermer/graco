#![feature(future_join, int_roundings)]

pub mod matching;

mod coarsen_graph;
pub use self::coarsen_graph::{CoarsenCounts, CoarsenGraph, CoarsenGraphInput, CoarsenGraphOutput};

mod counts_fallback;
