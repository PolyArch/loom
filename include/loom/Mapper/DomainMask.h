//===-- DomainMask.h - Domain-resource masking for ADGs -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Prunes unused functional nodes from a domain ADG before single-app mapping.
// Domain ADGs provision PEs for all apps in a domain; when mapping a single
// app, unused PEs consume switch ports and cause routing congestion. This
// utility removes functional nodes that are not compatible with the current
// DFG, freeing switch ports for routing.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_DOMAINMASK_H
#define LOOM_MAPPER_DOMAINMASK_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/TechMapper.h"

namespace loom {

/// Prune surplus functional nodes from a domain ADG before single-app mapping.
///
/// Domain ADGs provision PEs for all apps in a domain. When mapping a single
/// app, unused PEs consume switch ports and cause routing congestion. This
/// utility removes surplus functional nodes using coverage-safe greedy removal:
///
///   1. Run TechMapper to find all ADG candidates for each DFG node.
///   2. Remove functional nodes not compatible with any DFG node.
///   3. Greedily remove the least-useful compatible nodes while ensuring
///      every DFG node retains at least \p minCandidates candidates.
///
/// Memory nodes, module I/O sentinels, routing nodes, and virtual temporal
/// PE nodes are always retained.
///
/// The ADG graph is modified in-place. Callers should clone() the original
/// graph before calling this function if the original must be preserved.
void pruneDomainADG(Graph &adg, const Graph &dfg,
                    unsigned minCandidates = 2);

} // namespace loom

#endif // LOOM_MAPPER_DOMAINMASK_H
