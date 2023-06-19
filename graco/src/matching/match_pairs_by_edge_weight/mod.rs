use std::mem;

use empa::buffer;
use empa::buffer::Buffer;
use empa::command::CommandEncoder;
use empa::device::Device;
use empa::type_flag::{O, X};

use crate::matching::match_pairs_by_edge_weight::assign_node_colors::{
    AssignNodeColors, AssignNodeColorsInput,
};
use crate::matching::match_pairs_by_edge_weight::finalize_matching::FinalizeMatching;
use crate::matching::match_pairs_by_edge_weight::find_matches::{FindMatches, FindMatchesInput};
use crate::matching::match_pairs_by_edge_weight::make_proposals::{
    MakeProposals, MakeProposalsInput,
};
use crate::matching::match_pairs_by_edge_weight::match_state::MatchState;

mod assign_node_colors;
mod finalize_matching;
mod find_matches;
mod make_proposals;
mod match_state;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MatchPairsByEdgeWeightConfig {
    pub rounds: usize,
    pub prng_seed: u32,
}

impl Default for MatchPairsByEdgeWeightConfig {
    fn default() -> Self {
        MatchPairsByEdgeWeightConfig {
            rounds: 8,
            prng_seed: 1,
        }
    }
}

pub struct MatchPairsByEdgeWeightInput<'a, U0, U1, U2> {
    pub nodes_edge_offset: buffer::View<'a, [u32], U0>,
    pub nodes_edges: buffer::View<'a, [u32], U1>,
    pub nodes_edge_weights: buffer::View<'a, [f32], U2>,
}

pub struct MatchPairsByEdgeWeight {
    device: Device,
    assign_node_colors: AssignNodeColors,
    make_proposals: MakeProposals,
    find_matches: FindMatches,
    finalize_matching: FinalizeMatching,
    config: MatchPairsByEdgeWeightConfig,
    prng_seeds: Vec<Buffer<u32, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>>,
    has_live_nodes: Buffer<u32, buffer::Usages<O, O, X, X, O, O, X, O, O, O>>,
    proposals: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
}

impl MatchPairsByEdgeWeight {
    pub fn init(device: Device, config: MatchPairsByEdgeWeightConfig) -> Self {
        let assign_node_colors = AssignNodeColors::init(device.clone());
        let make_proposals = MakeProposals::init(device.clone());
        let find_matches = FindMatches::init(device.clone());
        let finalize_matching = FinalizeMatching::init(device.clone());

        let mut rng = oorandom::Rand32::new(1);
        let mut prng_seeds = Vec::with_capacity(config.rounds);

        for _ in 0..config.rounds {
            prng_seeds
                .push(device.create_buffer(rng.rand_u32(), buffer::Usages::uniform_binding()));
        }

        let has_live_nodes = device.create_buffer(
            0,
            buffer::Usages::storage_binding()
                .and_uniform_binding()
                .and_copy_dst(),
        );
        let proposals =
            device.create_slice_buffer_zeroed(1, buffer::Usages::storage_binding().and_copy_dst());

        MatchPairsByEdgeWeight {
            device,
            assign_node_colors,
            make_proposals,
            find_matches,
            finalize_matching,
            config,
            prng_seeds,
            has_live_nodes,
            proposals,
        }
    }

    pub fn encode<U0, U1, U2, U3>(
        &mut self,
        mut encoder: CommandEncoder,
        input: MatchPairsByEdgeWeightInput<U0, U1, U2>,
        nodes_match: buffer::View<[u32], U3>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
    {
        let MatchPairsByEdgeWeightInput {
            nodes_edge_offset,
            nodes_edges,
            nodes_edge_weights,
        } = input;

        let nodes_match_state: buffer::View<[MatchState], U3> =
            unsafe { mem::transmute(nodes_match) };

        if self.proposals.len() < nodes_edge_offset.len() {
            self.proposals = self
                .device
                .create_slice_buffer_zeroed(nodes_edge_offset.len(), self.proposals.usage());
        }

        encoder = encoder.clear_buffer_slice(self.proposals.view());

        for round in 0..self.config.rounds {
            encoder = encoder.clear_buffer(self.has_live_nodes.view());
            encoder = self.assign_node_colors.encode(
                encoder,
                AssignNodeColorsInput {
                    prng_seed: self.prng_seeds[round].view(),
                    nodes_match_state,
                    has_live_nodes: self.has_live_nodes.view(),
                },
            );
            encoder = self.make_proposals.encode(
                encoder,
                MakeProposalsInput {
                    has_live_nodes: self.has_live_nodes.view(),
                    nodes_match_state,
                    nodes_edge_offset,
                    nodes_edges,
                    nodes_edge_weights,
                    nodes_proposal: self.proposals.view(),
                },
            );
            encoder = self.find_matches.encode(
                encoder,
                FindMatchesInput {
                    has_live_nodes: self.has_live_nodes.view(),
                    nodes_match_state,
                    nodes_edge_offset,
                    nodes_edges,
                    nodes_edge_weights,
                    nodes_proposal: self.proposals.view(),
                },
            );
        }

        let nodes_match: buffer::View<[u32], U3> = unsafe { mem::transmute(nodes_match_state) };

        encoder = self.finalize_matching.encode(encoder, nodes_match);

        encoder
    }
}
