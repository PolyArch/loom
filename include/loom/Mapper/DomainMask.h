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
/// Two-phase pruning:
///   1. Functional pruning: Run TechMapper to find candidates, then remove
///      all functional PE nodes that are not compatible with any DFG
///      operation. Candidate PEs are always retained.
///   2. Routing pruning: BFS from retained endpoints (functional, memory,
///      module I/O) to find reachable routing nodes. Remove routing nodes
///      not reachable from any retained endpoint.
///
/// Memory nodes, module I/O sentinels, and virtual temporal PE nodes are
/// always retained.
///
/// \p minCandidates is reserved for future use (candidate-thinning pass).
///
/// The ADG graph is modified in-place. Callers should clone() the original
/// graph before calling this function if the original must be preserved.
void pruneDomainADG(Graph &adg, const Graph &dfg,
                    unsigned minCandidates = 2);

} // namespace loom

#endif // LOOM_MAPPER_DOMAINMASK_H
