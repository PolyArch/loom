#ifndef LOOM_LIB_MAPPING_VERIFIERSTATE_H
#define LOOM_LIB_MAPPING_VERIFIERSTATE_H

#include "FabricOccurrenceIndex.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::mapping::detail {

struct EndpointKey {
  bool actor;
  std::uint64_t owner;
  PortDirection direction;
  std::uint32_t index;

  friend bool operator==(const EndpointKey &lhs, const EndpointKey &rhs) {
    return lhs.actor == rhs.actor && lhs.owner == rhs.owner &&
           lhs.direction == rhs.direction && lhs.index == rhs.index;
  }
  friend bool operator<(const EndpointKey &lhs, const EndpointKey &rhs) {
    return std::tie(lhs.actor, lhs.owner, lhs.direction, lhs.index) <
           std::tie(rhs.actor, rhs.owner, rhs.direction, rhs.index);
  }
};

using EdgeKey = std::pair<EndpointKey, EndpointKey>;

struct DataflowPortInfo {
  std::uint64_t graph;
  EndpointKey key;
  const PortDescriptor *descriptor;
};

struct ResolvedDataflowEdge {
  DataflowPortInfo source;
  DataflowPortInfo target;
};

using ActorPortKey = std::pair<PortDirection, std::uint32_t>;

struct MemoryActorInfo {
  const CanonicalMemoryActorView *view;
  std::map<ActorPortKey, MemoryAccessPortRole> ports;
};

struct DataflowIndex {
  EntityKinds kinds;
  std::map<std::uint64_t, const GraphDescriptor *> graphs;
  std::map<std::uint64_t, const ActorDescriptor *> actors;
  std::map<std::uint64_t, const LogicalMemoryRootDescriptor *>
      logicalMemoryRoots;
  std::map<std::uint64_t, MemoryActorInfo> memoryActors;
  std::map<EdgeKey, std::size_t> edgesByKey;
  std::vector<ResolvedDataflowEdge> edges;
};

struct FabricNodeKey {
  ::loom::fabric::FabricFuNodeKind kind;
  std::uint64_t fu;
  std::uint64_t ordinal;

  friend bool operator==(FabricNodeKey lhs, FabricNodeKey rhs) {
    return lhs.kind == rhs.kind && lhs.fu == rhs.fu &&
           lhs.ordinal == rhs.ordinal;
  }
  friend bool operator<(FabricNodeKey lhs, FabricNodeKey rhs) {
    return std::tie(lhs.kind, lhs.fu, lhs.ordinal) <
           std::tie(rhs.kind, rhs.fu, rhs.ordinal);
  }
};

inline FabricNodeKey
nodeKey(const ::loom::fabric::FabricFuTemplateNodeRef &node) {
  return FabricNodeKey{node.node, node.fu.id(), node.ordinal};
}

struct FabricIndex {
  EntityKinds kinds;
  std::map<std::uint64_t, const FuDescriptor *> functionalUnits;
  std::map<FabricNodeKey, const FabricOpDescriptor *> operations;
  std::map<std::uint64_t, const MemoryServiceDomainDescriptor *>
      memoryServiceDomains;
  std::map<std::uint64_t, const MemoryImplementationDescriptor *>
      memoryImplementations;
  std::map<std::uint64_t, const MemoryOperationPortTemplateDescriptor *>
      memoryOperationPortTemplates;
  std::map<std::uint64_t, const MemoryInternalConnectionDescriptor *>
      memoryInternalConnections;
  std::map<std::uint64_t, const MemorySemanticEncodingDescriptor *>
      memorySemanticEncodings;
  std::shared_ptr<const ValidatedFabricProjection> projection;
};

struct RealizationActors {
  const ComputeRealizationDraft *record;
  std::uint64_t graph;
  std::map<std::uint64_t, const ActorDescriptor *> actors;
};

llvm::Expected<DataflowPortInfo>
resolveActorPortReference(const ActorPortRef &port,
                          const ArtifactIdentity &artifact,
                          const DataflowIndex &index);

llvm::Expected<std::vector<ValidatedComputeBoundaryPort>>
verifyComputeRealization(const RealizationActors &realization,
                         const DataflowProgramView &dataflow,
                         const DataflowIndex &dataflowIndex,
                         const FabricIndex &fabricIndex);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_VERIFIERSTATE_H
