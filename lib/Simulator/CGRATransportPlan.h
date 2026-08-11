#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTPLAN_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTPLAN_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace loom::sim::detail {

enum class CgraPhysicalUseClientKind : std::uint8_t {
  ComputeTransition,
  MemoryTransition,
  ProducedTransport,
  ConsumedTransport,
  TraversalTransport,
};

inline constexpr std::uint64_t invalidCgraTransportOrdinal =
    std::numeric_limits<std::uint64_t>::max();

enum class CgraTraversalStorageKind : std::uint8_t {
  None,
  BufferedFifo,
  RegisterFifoWrite,
  RegisterFifoRead,
};

struct CgraTraversalUsePlan final {
  ::loom::fabric::FabricUsePatternRef pattern;
  ::loom::fabric::FabricTraversalActivationGroupView activationGroup;
  std::uint64_t physicalUseOrdinal = invalidCgraTransportOrdinal;
};

struct CgraSelectedTraversalPlan final {
  ::loom::fabric::FabricPhysicalTraversalRef reference;
  ::loom::fabric::FabricPhysicalTraversalKind kind;
  CgraTraversalStorageKind storageKind = CgraTraversalStorageKind::None;
  std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t impliedUseOffset = 0;
  std::uint32_t impliedUseCount = 0;
};

struct CgraTraversalStoragePlan final {
  CgraTraversalStorageKind kind = CgraTraversalStorageKind::None;
  std::uint32_t capacity = 0;
  ::loom::fabric::FabricUsePatternRef enqueuePattern;
  ::loom::fabric::FabricUsePatternRef dequeuePattern;
  std::optional<::loom::fabric::FabricUsePatternRef> simultaneousPattern;
  std::uint64_t enqueuePhysicalUseOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t dequeuePhysicalUseOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t simultaneousPhysicalUseOrdinal = invalidCgraTransportOrdinal;
  bool independentReadWriteServices = false;
};

struct CgraPhysicalTagPlan final {
  llvm::APInt value = llvm::APInt(1, 0);
};

struct CgraRouteNodePlan final {
  std::uint32_t parentOrdinal = std::numeric_limits<std::uint32_t>::max();
  std::uint64_t incomingTraversalOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
};

struct CgraRouteSinkPlan final {
  ::dataflow::CanonicalGraphConsumerEndpointRef sink;
  std::uint32_t nodeOrdinal = 0;
  std::uint64_t localTraversalOrdinal = invalidCgraTransportOrdinal;
};

struct CgraRoutePlan final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::dataflow::GraphRef graph;
  std::uint64_t localTraversalOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t nodeOffset = 0;
  std::uint32_t nodeCount = 0;
  std::uint64_t sinkOffset = 0;
  std::uint32_t sinkCount = 0;
};

struct CgraLocalTransferSinkPlan final {
  ::dataflow::CanonicalGraphConsumerEndpointRef sink;
};

struct CgraLocalTransferPlan final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::dataflow::GraphRef graph;
  std::uint64_t sinkOffset = 0;
  std::uint32_t sinkCount = 0;
};

struct CgraProducedPhysicalUsePlan final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::uint64_t physicalUseOffset = 0;
  std::uint32_t physicalUseCount = 0;
};

struct CgraConsumedPhysicalUsePlan final {
  ::dataflow::CanonicalGraphConsumerEndpointRef consumer;
  std::uint64_t physicalUseOffset = 0;
  std::uint32_t physicalUseCount = 0;
};

/// Removable dense projection of the exact selected Spatial RouteTrees and
/// Fabric traversal contracts. Persistent references remain only in this cold
/// plan; dynamic execution indexes the flat arrays by ordinal.
struct CgraTransportPlan final {
  std::vector<CgraSelectedTraversalPlan> traversals;
  std::vector<CgraTraversalUsePlan> traversalUses;
  std::vector<CgraTraversalStoragePlan> traversalStorages;
  std::vector<CgraPhysicalTagPlan> physicalTags;
  std::vector<CgraRoutePlan> routes;
  std::vector<CgraRouteNodePlan> routeNodes;
  std::vector<CgraRouteSinkPlan> routeSinks;
  std::vector<CgraLocalTransferPlan> localTransfers;
  std::vector<CgraLocalTransferSinkPlan> localTransferSinks;
  std::vector<::dataflow::ActorTokenResultRef> discardedResults;
  std::vector<CgraProducedPhysicalUsePlan> producedUses;
  std::vector<CgraConsumedPhysicalUsePlan> consumedUses;
  std::vector<std::uint64_t> endpointPhysicalUses;
};

llvm::Expected<CgraTransportPlan> freezeCgraTransportPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<::dataflow::GraphRef> mappedGraphs,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTPLAN_H
