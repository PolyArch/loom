#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTPLAN_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTPLAN_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/IR/TemporalOperandBuffer.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
  /// Fabric-owned requester or ordinary UsePattern group used for resource
  /// arbitration. It is deliberately broader than the Mapping activation.
  ::loom::fabric::FabricTraversalRequesterGroupView requesterGroup;
  /// Dense Mapping-derived atomic activation identity. Spatial switch uses
  /// `(occurrence, input)`; Temporal switch uses `(row, input)`. Both may be
  /// finer than the Fabric requester used for arbitration or configuration.
  std::uint64_t activationInstanceOrdinal = invalidCgraTransportOrdinal;
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
  /// Dequeue scheduling discipline declared by the selected Fabric owner.
  ::fabric::FifoQueueDiscipline queueDiscipline =
      ::fabric::FifoQueueDiscipline::StrictFifo;
  ::loom::fabric::FabricUsePatternRef enqueuePattern;
  ::loom::fabric::FabricUsePatternRef dequeuePattern;
  std::optional<::loom::fabric::FabricUsePatternRef> simultaneousPattern;
  /// The internal arbitration transition of a per-tag virtual channel queue:
  /// one refused offer rotates the cursor at the cycle boundary. Absent for
  /// strict queues, which present only the physical head.
  std::optional<::loom::fabric::FabricUsePatternRef> offerAdvancePattern;
  std::uint64_t enqueuePhysicalUseOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t dequeuePhysicalUseOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t simultaneousPhysicalUseOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t offerAdvancePhysicalUseOrdinal = invalidCgraTransportOrdinal;
  bool independentReadWriteServices = false;
};

struct CgraPhysicalTagPlan final {
  llvm::APInt value = llvm::APInt(1, 0);
};

/// Interns the plan's Physical Tag values into virtual-channel ranks: equal
/// values share one rank regardless of which plan segment produced them, and
/// ranks follow the canonical ascending unsigned value order that a hardware
/// arbiter rotates through. A rank is a derived cache of the tag value, never
/// a semantic identity; a cold verifier recomputes this exact vector from the
/// plan and compares.
inline std::vector<std::uint32_t>
internPhysicalTagChannelRanks(llvm::ArrayRef<CgraPhysicalTagPlan> tags) {
  llvm::SmallVector<std::uint64_t, 16> order;
  order.reserve(tags.size());
  for (std::uint64_t ordinal = 0; ordinal != tags.size(); ++ordinal)
    order.push_back(ordinal);
  llvm::sort(order, [&](std::uint64_t lhs, std::uint64_t rhs) {
    return ::fabric::comparePhysicalTagValues(tags[lhs].value,
                                              tags[rhs].value) < 0;
  });
  std::vector<std::uint32_t> ranks(tags.size(), 0);
  std::uint32_t rank = 0;
  for (auto [position, ordinal] : llvm::enumerate(order)) {
    if (position != 0 &&
        ::fabric::comparePhysicalTagValues(tags[order[position - 1]].value,
                                           tags[ordinal].value) != 0)
      ++rank;
    ranks[ordinal] = rank;
  }
  return ranks;
}

struct CgraRouteNodePlan final {
  std::uint32_t parentOrdinal = std::numeric_limits<std::uint32_t>::max();
  std::uint64_t incomingTraversalOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t impliedUseOffset = invalidCgraTransportOrdinal;
  std::uint32_t impliedUseCount = 0;
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
  std::uint64_t writeTraversalOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t readTraversalOrdinal = invalidCgraTransportOrdinal;
  std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
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

struct CgraPeOperandQueueMatchPlan final {
  ::fabric::LogicalOperandQueueKey queue;
  ::loom::fabric::FabricFuOccurrenceRef fu;
  std::uint32_t allocationUnit = 0;
  std::uint32_t entryCapacity = 0;
  std::uint64_t consumerOffset = 0;
  std::uint32_t consumerCount = 0;
};

struct CgraPeOperandQueueConsumerPlan final {
  ::dataflow::CanonicalGraphConsumerEndpointRef consumer;
};

struct CgraPeOperandQueueActivationPlan final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::loom::fabric::FabricTransportEndpointRef ingress;
  llvm::APInt tag = llvm::APInt(1, 0);
  std::uint64_t matchOffset = 0;
  std::uint32_t matchCount = 0;
};

struct CgraPeOperandBufferPlan final {
  ::loom::fabric::FabricPeOccurrenceRef pe;
  ::fabric::OperandBufferMode mode{};
  std::uint32_t contextCount = 0;
  std::uint32_t entriesPerAllocationUnit = 0;
  std::vector<std::uint32_t> fuInputCounts;
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
  std::vector<CgraPeOperandQueueActivationPlan> operandQueueActivations;
  std::vector<CgraPeOperandBufferPlan> operandBuffers;
  std::vector<CgraPeOperandQueueMatchPlan> operandQueueMatches;
  std::vector<CgraPeOperandQueueConsumerPlan> operandQueueConsumers;
  ::loom::mapping::SpatialPeOperandProgressFeedback operandQueueProgress;
};

llvm::Expected<CgraTransportPlan> freezeCgraTransportPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<::dataflow::GraphRef> mappedGraphs,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTPLAN_H
