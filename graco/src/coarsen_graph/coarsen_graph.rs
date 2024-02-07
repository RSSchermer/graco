use std::future::join;

use empa::buffer;
use empa::buffer::{Buffer, Uniform};
use empa::command::{CommandEncoder, DispatchWorkgroups};
use empa::device::Device;
use empa::type_flag::{O, X};
use empa_tk::find_runs::{FindRuns, FindRunsInput, FindRunsOutput};
use empa_tk::gather_by::{GatherBy, GatherByInput};
use empa_tk::prefix_sum::{PrefixSum, PrefixSumInput};
use empa_tk::radix_sort::{RadixSortBy, RadixSortByInput};
use empa_tk::scatter_by::{ScatterBy, ScatterByInput};

use crate::coarsen_graph::collect_coarse_nodes_edge_weights::{
    CollectCoarseNodesEdgeWeights, CollectCoarseNodesEdgeWeightsResources,
};
use crate::coarsen_graph::compact_coarse_edges::{CompactCoarseEdges, CompactCoarseEdgesResources};
use crate::coarsen_graph::finalize_coarse_nodes_edge_offset::{
    FinalizeCoarseNodesEdgeOffset, FinalizeCoarseNodesEdgeOffsetResources,
};
use crate::coarsen_graph::gather_edge_owner_list::{
    GatherEdgeOwnerList, GatherEdgeOwnerListResources,
};
use crate::coarsen_graph::generate_dispatches::{GenerateDispatches, GenerateDispatchesResources};
use crate::coarsen_graph::generate_index_list::{GenerateIndexList, GenerateIndexListResources};
use crate::coarsen_graph::mark_coarse_edge_validity::{
    MarkCoarseEdgeValidity, MarkCoarseEdgeValidityResources,
};
use crate::coarsen_graph::resolve_coarse_edge_ref_count::{
    ResolveCoarseEdgeRefCount, ResolveCoarseEdgeRefCountResources,
};
use crate::coarsen_graph::DEFAULT_GROUP_SIZE;

pub struct CoarsenCounts {
    pub node_count: Uniform<u32>,
    pub edge_ref_count: Uniform<u32>,
}

pub struct CoarsenGraphInput<'a, U0, U1, U2, U3, U4, U5> {
    pub fine_nodes_edge_offset: buffer::View<'a, [u32], U0>,
    pub fine_nodes_edges: buffer::View<'a, [u32], U1>,
    pub fine_nodes_edge_weights: buffer::View<'a, [u32], U2>,
    pub fine_nodes_matching: buffer::View<'a, [u32], U3>,
    pub temporary_storage_0: buffer::View<'a, [u32], U4>,
    pub temporary_storage_1: buffer::View<'a, [u32], U5>,
    pub counts: Option<CoarsenCounts>,
}

pub struct CoarsenGraphOutput<'a, U0, U1, U2, U3, U4, U5, U6, U7> {
    pub fine_nodes_mapping: buffer::View<'a, [u32], U0>,
    pub coarse_nodes_mapping_offset: buffer::View<'a, [u32], U1>,
    pub coarse_nodes_mapping: buffer::View<'a, [u32], U2>,
    pub coarse_node_count: buffer::View<'a, u32, U3>,
    pub coarse_edge_ref_count: buffer::View<'a, u32, U4>,
    pub coarse_nodes_edge_offset: buffer::View<'a, [u32], U5>,
    pub coarse_nodes_edges: buffer::View<'a, [u32], U6>,
    pub coarse_nodes_edge_weights: buffer::View<'a, [u32], U7>,
}

pub struct CoarsenGraph {
    device: Device,
    generate_dispatches: GenerateDispatches,
    generate_index_list: GenerateIndexList,
    gather_edge_owner_list: GatherEdgeOwnerList,
    mark_coarse_edge_validity: MarkCoarseEdgeValidity,
    collect_coarse_nodes_edge_weights: CollectCoarseNodesEdgeWeights,
    compact_coarse_edges: CompactCoarseEdges,
    resolve_coarse_edge_ref_count: ResolveCoarseEdgeRefCount,
    finalize_coarse_nodes_edge_offset: FinalizeCoarseNodesEdgeOffset,
    sort_by: RadixSortBy<u32, u32>,
    find_runs: FindRuns<u32>,
    scatter_by: ScatterBy<u32, u32>,
    gather_by: GatherBy<u32, u32>,
    prefix_sum_inclusive: PrefixSum<u32>,
    group_size: Buffer<u32, buffer::Usages<O, O, O, X, O, O, O, O, O, O>>,
    node_count_dispatch: Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
    edge_ref_count_dispatch:
        Buffer<DispatchWorkgroups, buffer::Usages<O, X, X, O, O, O, O, O, O, O>>,
}

impl CoarsenGraph {
    pub async fn init(device: Device) -> Self {
        let (
            generate_dispatches,
            generate_index_list,
            gather_edge_owner_list,
            mark_coarse_edge_validity,
            collect_coarse_nodes_edge_weights,
            compact_coarse_edges,
            resolve_coarse_edge_ref_count,
            finalize_coarse_nodes_edge_offset,
            sort_by,
            find_runs,
            scatter_by,
            gather_by,
            prefix_sum_inclusive,
        ) = join!(
            GenerateDispatches::init(device.clone()),
            GenerateIndexList::init(device.clone()),
            GatherEdgeOwnerList::init(device.clone()),
            MarkCoarseEdgeValidity::init(device.clone()),
            CollectCoarseNodesEdgeWeights::init(device.clone()),
            CompactCoarseEdges::init(device.clone()),
            ResolveCoarseEdgeRefCount::init(device.clone()),
            FinalizeCoarseNodesEdgeOffset::init(device.clone()),
            RadixSortBy::init_u32(device.clone()),
            FindRuns::init_u32(device.clone()),
            ScatterBy::init_u32(device.clone()),
            GatherBy::init_u32(device.clone()),
            PrefixSum::init_inclusive_u32(device.clone()),
        )
        .await;

        let group_size =
            device.create_buffer(DEFAULT_GROUP_SIZE, buffer::Usages::uniform_binding());
        let node_count_dispatch = device.create_buffer(
            DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );
        let edge_ref_count_dispatch = device.create_buffer(
            DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            },
            buffer::Usages::storage_binding().and_indirect(),
        );

        CoarsenGraph {
            device,
            generate_dispatches,
            generate_index_list,
            gather_edge_owner_list,
            mark_coarse_edge_validity,
            collect_coarse_nodes_edge_weights,
            compact_coarse_edges,
            resolve_coarse_edge_ref_count,
            finalize_coarse_nodes_edge_offset,
            sort_by,
            find_runs,
            scatter_by,
            gather_by,
            prefix_sum_inclusive,
            group_size,
            node_count_dispatch,
            edge_ref_count_dispatch,
        }
    }

    pub fn encode<U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13>(
        &mut self,
        mut encoder: CommandEncoder,
        input: CoarsenGraphInput<U0, U1, U2, U3, U4, U5>,
        output: CoarsenGraphOutput<U6, U7, U8, U9, U10, U11, U12, U13>,
    ) -> CommandEncoder
    where
        U0: buffer::StorageBinding,
        U1: buffer::StorageBinding,
        U2: buffer::StorageBinding,
        U3: buffer::StorageBinding,
        U4: buffer::StorageBinding + buffer::CopyDst + 'static,
        U5: buffer::StorageBinding + buffer::CopyDst + 'static,
        U6: buffer::StorageBinding,
        U7: buffer::StorageBinding,
        U8: buffer::StorageBinding,
        U9: buffer::StorageBinding,
        U10: buffer::StorageBinding,
        U11: buffer::StorageBinding,
        U12: buffer::StorageBinding,
        U13: buffer::StorageBinding + buffer::CopyDst + 'static,
    {
        // This coarsening algorithm is based on the algorithm described by Auer et al. "Graph
        // Coarsening and Clustering on the GPU", though it deviates in how it constructs the
        // coarse edge list. Though Auer et al. don't go into detail in the article on how they
        // "compress edges", from the accompanying git repository
        // (https://github.com/BasFaggingerAuer/Multicore-Clustering), this seems to involve
        // assigning 1 thread per coarse node to first heap-sort that nodes edges, then filter out
        // duplicates and self-references, and then sum the edge weights together. For unbalanced
        // graphs, this might lead to poor occupancy.
        //
        // The method we use here involves device-wide radix sorting and prefix-sum based compaction
        // to achieve the same. This comes at a cost of requiring additional temporary storage
        // memory, but will achieve good device utilization, regardless of how poorly balanced the
        // graph is.

        let CoarsenGraphInput {
            fine_nodes_edge_offset,
            fine_nodes_edges,
            fine_nodes_edge_weights,
            fine_nodes_matching,
            temporary_storage_0,
            temporary_storage_1,
            counts,
        } = input;

        let CoarsenGraphOutput {
            fine_nodes_mapping,
            coarse_nodes_mapping_offset,
            coarse_nodes_mapping,
            coarse_node_count,
            coarse_edge_ref_count,
            coarse_nodes_edge_offset,
            coarse_nodes_edges,
            coarse_nodes_edge_weights,
        } = output;

        let dispatch_indirect = counts.is_some();

        if let Some(counts) = counts.as_ref() {
            encoder = self.generate_dispatches.encode(
                encoder,
                GenerateDispatchesResources {
                    group_size: self.group_size.uniform(),
                    node_count: counts.node_count.clone(),
                    edge_ref_count: counts.edge_ref_count.clone(),
                    node_count_dispatch: self.node_count_dispatch.storage(),
                    edge_ref_count_dispatch: self.edge_ref_count_dispatch.storage(),
                },
            );
        }

        let fallback_node_count = fine_nodes_edge_offset.len() as u32;
        let fallback_edge_ref_count = fine_nodes_edges.len() as u32;

        let (node_count, edge_ref_count) = counts
            .as_ref()
            .map(|c| (c.node_count.clone(), c.edge_ref_count.clone()))
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

        encoder = self.generate_index_list.encode(
            encoder,
            GenerateIndexListResources {
                count: node_count.clone(),
                data: coarse_nodes_mapping.storage(),
            },
            dispatch_indirect,
            self.node_count_dispatch.view(),
            fallback_node_count,
        );

        encoder = self.sort_by.encode(
            encoder,
            RadixSortByInput {
                keys: fine_nodes_matching,
                values: coarse_nodes_mapping,
                temporary_key_storage: temporary_storage_0,
                temporary_value_storage: temporary_storage_1,
                count: Some(node_count.clone()),
            },
        );

        let run_mapping = temporary_storage_0;

        encoder = self.find_runs.encode(
            encoder,
            FindRunsInput {
                data: fine_nodes_matching,
                count: Some(node_count.clone()),
            },
            FindRunsOutput {
                run_count: coarse_node_count,
                run_starts: coarse_nodes_mapping_offset,
                run_mapping,
            },
        );

        encoder = self.scatter_by.encode(
            encoder,
            ScatterByInput {
                scatter_by: coarse_nodes_mapping,
                data: run_mapping,
                count: Some(node_count.clone()),
            },
            fine_nodes_mapping,
        );

        // We now have both a mapping from fine nodes to coarse nodes (`fine_nodes_mapping`) and
        // a mapping from coarse nodes to fine nodes (`coarse_nodes_mapping_offset` in combination
        // with `coarse_nodes_mapping`); we're now ready to construct the edge lists.

        // We'll need 4 `u32` slice buffers for this section that each have the capacity to store
        // the coarse edge count. To save the user a buffer allocation, we'll use the
        // `coarse_nodes_edge_weights` buffer as one of these buffers. These buffers will end up
        // being reused various times throughout this process, which makes it somewhat difficult to
        // track whats being stored where at any point; such is the price of trying to do this with
        // minimal memory allocation. At the end of this dance, the final step needs 2 helper
        // buffers to generate the final `coarse_nodes_edges` and `coarse_nodes_edge_weights`
        // buffers, which means that the helper data needs to end up in `temporary_storage_0` and
        // `temporary_storage_1`. These buffer aliases are arranged such that this works out.
        let storage_0 = coarse_nodes_edge_weights;
        let storage_1 = temporary_storage_0;
        let storage_2 = coarse_nodes_edges;
        let storage_3 = temporary_storage_1;

        // As a first step, we'll take the edge pointers from the fine level, and replace all node
        // indices pointed to with the indices of the nodes they map to in the coarse level, using
        // the `fine_nodes_mapping`.
        encoder = self.gather_by.encode(
            encoder,
            GatherByInput {
                gather_by: fine_nodes_edges,
                data: fine_nodes_mapping,
                count: Some(edge_ref_count.clone()),
            },
            storage_0,
        );

        // We now generate a new list of consecutive indices `0, 1, 2, ...`. We'll use this list to
        // do a "compound sort":

        encoder = self.generate_index_list.encode(
            encoder,
            GenerateIndexListResources {
                count: edge_ref_count.clone(),
                data: storage_1.storage(),
            },
            dispatch_indirect,
            self.edge_ref_count_dispatch.view(),
            fallback_edge_ref_count,
        );

        //  We'll first radix-sort this index list using mapped edge list we produced in the
        //  previous step. The index list now "stores the sort".

        encoder = self.sort_by.encode(
            encoder,
            RadixSortByInput {
                keys: storage_0,
                values: storage_1,
                temporary_key_storage: storage_2,
                temporary_value_storage: storage_3,
                count: Some(edge_ref_count.clone()),
            },
        );

        // We generate a new list based on the edge list from the fine level, where for each
        // fine level edge we store the index of the would-be "owner node" in the coarse level.
        // We "apply the sort" stored in the index list to this new list through a gather-by
        // operation.

        encoder = self.gather_edge_owner_list.encode(
            encoder,
            GatherEdgeOwnerListResources {
                fine_node_count: node_count.clone(),
                fine_edge_count: edge_ref_count.clone(),
                fine_nodes_edge_offset: fine_nodes_edge_offset.read_only_storage(),
                fine_nodes_mapping: fine_nodes_mapping.read_only_storage(),
                coarsened_edge_owner_list: storage_2.storage(),
            },
            dispatch_indirect,
            self.node_count_dispatch.view(),
            fallback_node_count,
        );

        encoder = self.gather_by.encode(
            encoder,
            GatherByInput {
                gather_by: storage_1,
                data: storage_2,
                count: Some(edge_ref_count.clone()),
            },
            storage_0,
        );

        // We radix-sort the index list again, now using the gathered owner list we created in the
        // previous step as the sort keys.

        encoder = self.sort_by.encode(
            encoder,
            RadixSortByInput {
                keys: storage_0,
                values: storage_1,
                temporary_key_storage: storage_2,
                temporary_value_storage: storage_3,
                count: Some(edge_ref_count.clone()),
            },
        );

        // Recreate the mapped edge list (it got "destroyed" by the first sort operation).

        encoder = self.gather_by.encode(
            encoder,
            GatherByInput {
                gather_by: fine_nodes_edges,
                data: fine_nodes_mapping,
                count: Some(edge_ref_count.clone()),
            },
            storage_2,
        );

        // We now apply the "compound sort" stored in the index list to this mapped edge list with
        // another gather-by operation.

        encoder = self.gather_by.encode(
            encoder,
            GatherByInput {
                gather_by: storage_1,
                data: storage_2,
                count: Some(edge_ref_count.clone()),
            },
            storage_3,
        );

        // We'll also apply it to the `fine_nodes_edge_weights` list to create a corresponding edge
        // weight list.

        encoder = self.gather_by.encode(
            encoder,
            GatherByInput {
                gather_by: storage_1,
                data: fine_nodes_edge_weights,
                count: Some(edge_ref_count.clone()),
            },
            storage_2,
        );

        // This first phase of constructing the coarsened edge list leverages the fact that radix
        // sort is a stable, order preserving sort. We have essentially applied a "compound sort" to
        // the mapped edge list. This new list has 2 interesting properties:
        //
        // 1. As a result of the second sort, the mapped edges in the list are grouped by the coarse
        //   nodes they belong to. These groups are also ordered by coarse node index (e.g. the
        //   first section of the list contains all the edges that belong to coarse node `0`, the
        //   second section of the list contains all the edges that belong to coarse node `1`,
        //   etc.).
        // 2. As a result of the first sort, within each group the edges are sorted by the index of
        //   the coarse node they point to.
        //
        // This, however, is still not what we need for the final coarse edge list for 2 reasons:
        //
        // - The edge list for each coarse node may contain self-references, if the coarse node is
        //   not the offspring of an unmatched node. That is because currently, the coarse node edge
        //   list merged the mapped edge lists of the of the 2 fine nodes that were matched. These 2
        //   fine nodes must have been adjacent nodes for them to be a valid match and thus each
        //   node will have pointed to the other node. These 2 edge pointers have now become
        //   self-references.
        // - The edge list may contain duplicates, if the coarse node is not the offspring of an
        //   unmatched node. This is because the 2 matched fine nodes may have both pointed to the
        //   same node, or two different fine nodes that have merged into 1 coarse node.
        //
        // We will have to filter out these self-referencing edges and duplicate edges to arrive
        // at the correct final coarse edge list. We'll produce a helper list that stores `0` for an
        // invalid edge and `1` for valid a valid edge.
        //
        // Conveniently, to help us construct this "validity list" we can use the fact that the
        // second sort also produced a list that maps each edge to its owner node, since it also
        // sorted the keys (this list is currently stored in `temporary_storage_0`). We can
        // therefore very quickly check if an edge is self-referencing. Additionally, as a result
        // of property 2. of the sorted coarse edge list, duplicate edges will all be stored
        // consecutively, and thus we only need to inspect the previous element in the list to
        // determine if an edge is a duplicate. An edge is therefore valid if:
        //
        // - Looking up the owner node does not produce the same node index as the edge's target
        //   node index.
        // - Looking up both the previous edge and the owner node of the previous edge does not
        //   produce the same `(owner node, referenced node)` pair as the current edge and its
        //   owner node.

        // We'll first "find the runs" in the owner node list, these run offsets are not the final
        // edge offsets for the coarse level, but we'll convert them later.
        encoder = self.find_runs.encode(
            encoder,
            FindRunsInput {
                data: storage_0,
                count: Some(edge_ref_count.clone()),
            },
            FindRunsOutput {
                run_count: coarse_node_count,
                run_starts: coarse_nodes_edge_offset,
                run_mapping: storage_1,
            },
        );

        // Now construct the validity list. In addition to storing the validity state for each edge
        // in a separate validity list, will also store it in the 2 most significant bits of the
        // mapped edge list. We do this because the validity list will be "destroyed" (by a prefix
        // sum operation), but we'd still like to use the validity information after that, without
        // having to recompute it or requiring an additional storage buffer. We use 2 bits rather
        // than 1, because we'd also like to know the reason for the invalidity (whether it is a
        // duplicate edge, or a self-referencing edge); this simplifies computing the
        // `coarse_node_edge_weights` later (we *do* want to combine duplicate edge weights into the
        // new edge weight, we *do not* want to combine self-referencing edge weights into any new
        // edge weights). The 2 most significant bits will carry the following meaning:
        //
        // 0 - invalid due to being a self-referencing edge
        // 1 - invalid due to being a duplicate edge
        // 2 - valid
        //
        // Note that this restricts the maximum graph size to 2^30 nodes.
        encoder = self.mark_coarse_edge_validity.encode(
            encoder,
            MarkCoarseEdgeValidityResources {
                count: edge_ref_count.clone(),
                owner_nodes: storage_0.read_only_storage(),
                mapped_edges: storage_3.storage(),
                validity: storage_1.storage(),
            },
            dispatch_indirect,
            self.edge_ref_count_dispatch.view(),
            fallback_edge_ref_count,
        );

        // We now perform an inclusive prefix-sum operation over the validity list. After this, for
        // each valid edge, subtracting `1` from the value in this list will give the index of the
        // position to copy the mapped edge to in the final compacted `coarse_nodes_edges` list.
        //
        // This index also identifies the coarse edge weight position to sum edge weights into, both
        // for valid edges and invalid-but-not-self-referencing edges.
        encoder = self.prefix_sum_inclusive.encode(
            encoder,
            PrefixSumInput {
                data: storage_1,
                count: Some(edge_ref_count.clone()),
            },
        );

        // Compute the final `coarse_nodes_edge_weights` by (atomically) adding together the
        // weights for the mapped edges to the index provided by the validity prefix-sum, except
        // if edge is marked as invalid due to self-referencing.
        encoder = encoder.clear_buffer_slice(storage_0);
        encoder = self.collect_coarse_nodes_edge_weights.encode(
            encoder,
            CollectCoarseNodesEdgeWeightsResources {
                count: edge_ref_count.clone(),
                mapped_edges: storage_3.read_only_storage(),
                mapped_edge_weights: storage_2.read_only_storage(),
                validity_prefix_sum: storage_1.read_only_storage(),
                coarse_nodes_edge_weights: storage_0.storage(),
            },
            dispatch_indirect,
            self.edge_ref_count_dispatch.view(),
            fallback_edge_ref_count,
        );

        // Copy the edges from the uncompacted mapped edge list to their final positions in the
        // compacted `coarse_nodes_edges` list, if the edge is marked as "valid". Take care to not
        // copy to 2 most significant that store the validity state.
        encoder = self.compact_coarse_edges.encode(
            encoder,
            CompactCoarseEdgesResources {
                count: edge_ref_count.clone(),
                mapped_edges: storage_3.read_only_storage(),
                validity_prefix_sum: storage_1.read_only_storage(),
                coarse_nodes_edges: storage_2.storage(),
            },
            dispatch_indirect,
            self.edge_ref_count_dispatch.view(),
            fallback_edge_ref_count,
        );

        // Resolve the edge ref count for the coarse level by copying the last number in the
        // validity prefix-sum list
        encoder = self.resolve_coarse_edge_ref_count.encode(
            encoder,
            ResolveCoarseEdgeRefCountResources {
                fine_level_edge_ref_count: edge_ref_count,
                validity_prefix_sum: storage_1.read_only_storage(),
                coarse_level_edge_ref_count: coarse_edge_ref_count.storage(),
            },
        );

        // Finally, we collected a preliminary version of `coarse_nodes_edge_offset` earlier, but
        // the offsets it currently contains point to the starts of the uncompacted node edge
        // ranges. We can adjust it by using the validity prefix-sum as a mapping to find the
        // offsets of the compacted node edge ranges. Note that the first of a node's edges in the
        // uncompacted range can be an invalid edge reference, in which case we add `1`.
        encoder = self.finalize_coarse_nodes_edge_offset.encode(
            encoder,
            FinalizeCoarseNodesEdgeOffsetResources {
                count: node_count,
                mapped_edges: storage_3.read_only_storage(),
                validity_prefix_sum: storage_1.read_only_storage(),
                coarse_nodes_edge_offset: coarse_nodes_edge_offset.storage(),
            },
            dispatch_indirect,
            self.node_count_dispatch.view(),
            fallback_node_count,
        );

        // And we're done!

        encoder
    }
}
