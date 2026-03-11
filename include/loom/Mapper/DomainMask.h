//===-- DomainMask.h - Domain-resource masking for ADGs -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Prunes unused resources from a domain ADG before single-app mapping.
// Domain ADGs provision PEs for all apps in a domain; when mapping a single
// app, unused PEs consume switch ports and cause routing congestion. This
// utility removes functional nodes not compatible with the current DFG and
// then removes routing nodes no longer reachable from retained endpoints.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_DOMAINMASK_H
#define LOOM_MAPPER_DOMAINMASK_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/TechMapper.h"

namespace loom {

/// Prune unused resources from a domain ADG before single-app mapping.
///
/// Endpoint-driven mask builder:
///   1. Run TechMapper to compute the full candidate universe (DFG -> ADG
///      node compatibility) for both functional PEs and memory nodes.
///   2. Partition candidate PEs into compatibility classes (PEs with
///      identical DFG-operation capability sets are interchangeable).
///   3. Compute demand per class using fractional allocation: each DFG
///      operation distributes 1 unit of demand across its compatible
///      classes. Retain ceil(demand) + (minCandidates-1) PEs per class.
///   4. Remove all functional PEs not retained (both non-candidates and
///      surplus candidates beyond computed demand).
///   5. Remove memory nodes not referenced by any DFG candidate. Domain
///      ADGs provision memory groups for all apps; surplus memory nodes
///      with high port counts consume switch capacity needed for routing.
///   6. Steiner-tree routing pruning: approximate a minimum subgraph
///      connecting retained endpoints via Voronoi-boundary BFS, then
///      expand by a buffer for routing redundancy.
///   7. Iteratively prune dead-end routing nodes: routing nodes with no
///      endpoint connections and at most one routing neighbor are removed.
///
/// Module I/O sentinels and virtual temporal PE nodes with retained FU
/// sub-nodes are always retained.
///
/// \p minCandidates controls deterministic slack: each compatibility class
///    retains at least ceil(demand) + (minCandidates - 1) PEs.
///
/// The ADG graph is modified in-place. Callers should clone() the original
/// graph before calling this function if the original must be preserved.
void pruneDomainADG(Graph &adg, const Graph &dfg,
                    unsigned minCandidates = 2);

} // namespace loom

#endif // LOOM_MAPPER_DOMAINMASK_H
