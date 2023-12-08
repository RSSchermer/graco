use std::mem;

use empa::buffer;
use empa::buffer::{Buffer, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups};
use empa::device::Device;
use empa::type_flag::{O, X};

use crate::matching::match_pairs_by_edge_weight::assign_node_colors::{
    AssignNodeColors, AssignNodeColorsResources,
};
use crate::matching::match_pairs_by_edge_weight::finalize_matching::{
    FinalizeMatching, FinalizeMatchingResources,
};
use crate::matching::match_pairs_by_edge_weight::find_matches::{
    FindMatches, FindMatchesResources,
};
use crate::matching::match_pairs_by_edge_weight::generate_dispatch::{
    GenerateDispatch, GenerateDispatchResources,
};
use crate::matching::match_pairs_by_edge_weight::make_proposals::{
    MakeProposals, MakeProposalsResources,
};
use crate::matching::match_pairs_by_edge_weight::match_state::MatchState;

mod assign_node_colors;
mod finalize_matching;
mod find_matches;
mod generate_dispatch;
mod make_proposals;
mod match_state;

pub const GROUP_SIZE: u32 = 256;

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

pub struct MatchPairsByEdgeWeightsCounts {
    pub node_count: Uniform<u32>,
    pub edge_ref_count: Uniform<u32>,
}

pub struct MatchPairsByEdgeWeightInput<'a, U0, U1, U2> {
    pub nodes_edge_offset: buffer::View<'a, [u32], U0>,
    pub nodes_edges: buffer::View<'a, [u32], U1>,
    pub nodes_edge_weights: buffer::View<'a, [u32], U2>,
    pub count: Option<MatchPairsByEdgeWeightsCounts>,
}

pub struct MatchPairsByEdgeWeight {
    device: Device,
    generate_dispatch: GenerateDispatch,
    assign_node_colors: AssignNodeColors,
    make_proposals: MakeProposals,
    find_matches: FindMatches,
    finalize_matching: FinalizeMatching,
    config: MatchPairsByEdgeWeightConfig,
    prng_seeds: Vec<Buffer<u32, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>>,
    has_live_nodes: Buffer<u32, buffer::Usages<O, O, X, X, O, O, X, O, O, O>>,
    group_size: Buffer<u32, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>,
    dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
    proposals: Buffer<[u32], buffer::Usages<O, O, X, O, O, O, X, O, O, O>>,
}

impl MatchPairsByEdgeWeight {
    pub fn init(device: Device, config: MatchPairsByEdgeWeightConfig) -> Self {
        let generate_dispatch = GenerateDispatch::init(device.clone());
        let assign_node_colors = AssignNodeColors::init(device.clone());
        let make_proposals = MakeProposals::init(device.clone());
        let find_matches = FindMatches::init(device.clone());
        let finalize_matching = FinalizeMatching::init(device.clone());

        let mut rng = oorandom::Rand32::new(config.prng_seed as u64);
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
        let group_size = device.create_buffer(GROUP_SIZE, buffer::Usages::uniform_binding());
        let dispatch = device.create_buffer(
            DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );
        let proposals =
            device.create_slice_buffer_zeroed(1, buffer::Usages::storage_binding().and_copy_dst());

        MatchPairsByEdgeWeight {
            device,
            generate_dispatch,
            assign_node_colors,
            make_proposals,
            find_matches,
            finalize_matching,
            config,
            prng_seeds,
            has_live_nodes,
            group_size,
            dispatch,
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
            count,
        } = input;

        let nodes_match_state: buffer::View<[MatchState], U3> =
            unsafe { mem::transmute(nodes_match) };

        if self.proposals.len() < nodes_edge_offset.len() {
            self.proposals = self
                .device
                .create_slice_buffer_zeroed(nodes_edge_offset.len(), self.proposals.usage());
        }

        let dispatch_indirect = count.is_some();

        let fallback_node_count = nodes_edge_offset.len() as u32;
        let fallback_edge_ref_count = nodes_edges.len() as u32;

        let (node_count, edge_ref_count) = count
            .map(|c| (c.node_count, c.edge_ref_count))
            .unwrap_or_else(|| {
                let node_count = self
                    .device
                    .create_buffer(fallback_node_count, buffer::Usages::uniform_binding())
                    .uniform();
                let edge_ref_count = self
                    .device
                    .create_buffer(fallback_edge_ref_count, buffer::Usages::uniform_binding())
                    .uniform();

                (node_count, edge_ref_count)
            });

        if dispatch_indirect {
            encoder = self.generate_dispatch.encode(
                encoder,
                GenerateDispatchResources {
                    group_size: self.group_size.uniform(),
                    count: node_count.clone(),
                    dispatch: self.dispatch.storage(),
                },
            );
        }

        encoder = encoder.clear_buffer_slice(self.proposals.view());

        for round in 0..self.config.rounds {
            encoder = encoder.clear_buffer(self.has_live_nodes.view());
            encoder = self.assign_node_colors.encode(
                encoder,
                AssignNodeColorsResources {
                    count: node_count.clone(),
                    prng_seed: self.prng_seeds[round].uniform(),
                    nodes_match_state: nodes_match_state.storage(),
                    has_live_nodes: self.has_live_nodes.storage(),
                },
                dispatch_indirect,
                self.dispatch.view(),
                fallback_node_count,
            );
            encoder = self.make_proposals.encode(
                encoder,
                MakeProposalsResources {
                    node_count: node_count.clone(),
                    edge_ref_count: edge_ref_count.clone(),
                    has_live_nodes: self.has_live_nodes.uniform(),
                    nodes_match_state: nodes_match_state.storage(),
                    nodes_edge_offset: nodes_edge_offset.read_only_storage(),
                    nodes_edges: nodes_edges.read_only_storage(),
                    nodes_edge_weights: nodes_edge_weights.read_only_storage(),
                    nodes_proposal: self.proposals.storage(),
                },
                dispatch_indirect,
                self.dispatch.view(),
                fallback_node_count,
            );
            encoder = self.find_matches.encode(
                encoder,
                FindMatchesResources {
                    node_count: node_count.clone(),
                    edge_ref_count: edge_ref_count.clone(),
                    has_live_nodes: self.has_live_nodes.uniform(),
                    nodes_match_state: nodes_match_state.storage(),
                    nodes_edge_offset: nodes_edge_offset.read_only_storage(),
                    nodes_edges: nodes_edges.read_only_storage(),
                    nodes_edge_weights: nodes_edge_weights.read_only_storage(),
                    nodes_proposal: self.proposals.read_only_storage(),
                },
                dispatch_indirect,
                self.dispatch.view(),
                fallback_node_count,
            );
        }

        let nodes_match: buffer::View<[u32], U3> = unsafe { mem::transmute(nodes_match_state) };

        encoder = self.finalize_matching.encode(
            encoder,
            FinalizeMatchingResources {
                count: node_count.clone(),
                nodes_match_state: nodes_match.storage(),
            },
            dispatch_indirect,
            self.dispatch.view(),
            fallback_node_count,
        );

        encoder
    }
}
