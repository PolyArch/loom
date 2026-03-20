#ifndef FCC_MAPPER_MAPPERROUTINGINTERNAL_H
#define FCC_MAPPER_MAPPERROUTINGINTERNAL_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace fcc {
namespace routing_detail {

/// Returns true if the node has resource_class == "routing".
bool isRoutingNode(const Node *node);

/// Returns true if portId is an output port of a multi-port routing crossbar.
bool isRoutingCrossbarOutputPort(IdIndex portId, const Graph &adg);

/// Returns true if opName is a software memory interface operation.
bool isSoftwareMemoryInterfaceOpName(llvm::StringRef opName);

/// Returns true if portId is an output port of a non-routing node.
bool isNonRoutingOutputPort(IdIndex portId, const Graph &adg);

/// Returns the locked first-hop input port for a non-routing source output
/// port, or INVALID_ID if no existing route from this source exists.
IdIndex getLockedFirstHopForSource(IdIndex swEdgeId, IdIndex srcHwPort,
                                   const MappingState &state);

struct TaggedPathObservation {
  MappingState::TaggedObservationKind kind =
      MappingState::TaggedObservationKind::RoutingOutput;
  IdIndex first = INVALID_ID;
  IdIndex second = INVALID_ID;
  uint64_t tag = 0;
};

struct TemporalSwitchTagRouteObservation {
  IdIndex nodeId = INVALID_ID;
  IdIndex inPortId = INVALID_ID;
  IdIndex outPortId = INVALID_ID;
  uint64_t tag = 0;
};

struct RuntimeTagPathFailure {
  size_t pathIndex = 0;
  IdIndex portId = INVALID_ID;
  uint64_t tag = 0;
  unsigned tagWidth = 0;
};

llvm::SmallVector<IdIndex, 16>
buildExportPathForEdge(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> rawPath,
                       const MappingState &state, const Graph &dfg,
                       const Graph &adg);

std::optional<RuntimeTagPathFailure> findRuntimeTagPathFailure(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> fullPath,
    const MappingState &state, const Graph &dfg, const Graph &adg);

void appendTaggedPathObservations(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> path, const MappingState &state,
    const Graph &dfg, const Graph &adg,
    llvm::SmallVectorImpl<TaggedPathObservation> &observations);

void appendTemporalSwitchTagRouteObservations(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> path, const MappingState &state,
    const Graph &dfg, const Graph &adg,
    llvm::SmallVectorImpl<TemporalSwitchTagRouteObservation> &observations);

bool observationsConflict(const TaggedPathObservation &lhs,
                          const TaggedPathObservation &rhs);

bool temporalSwitchTagRouteConflict(
    const TemporalSwitchTagRouteObservation &lhs,
    const TemporalSwitchTagRouteObservation &rhs);

} // namespace routing_detail
} // namespace fcc

#endif // FCC_MAPPER_MAPPERROUTINGINTERNAL_H
