#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTGRAPH_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTGRAPH_H

#include "CGRAExecutionPlan.h"
#include "DFGSimulatorInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::sim::detail {

/// Immutable graph-specific transport projection retained by the prepared
/// execution. Every binding derives from the exact frozen Mapping and semantic
/// graph; token state and initial channel occupancy remain per invocation.
struct CgraTransportGraph final {
  enum class SinkKind : std::uint8_t { Channel, Observation };

  struct SinkBinding final {
    SinkKind kind = SinkKind::Channel;
    ChannelOrdinal channel = 0;
    mlir::Value observation;
    std::uint64_t physicalUseOffset = 0;
    std::uint32_t physicalUseCount = 0;
    std::uint32_t consumedLocalActionOffset = 0;
    std::uint64_t operandQueueBinding = invalidCgraTransportOrdinal;
    std::uint64_t operandActivationOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t publicationBinding = invalidCgraTransportOrdinal;
    std::uint64_t semanticActorOrdinal = invalidCgraTransportOrdinal;
    std::uint32_t inputOrdinal = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t traversalTerminalCount = 0;
  };

  struct PublicationBinding final {
    struct OperandIngressQuery final {
      std::uint64_t buffer = invalidCgraTransportOrdinal;
      llvm::SmallVector<std::uint32_t, 8> matched;
      llvm::SmallVector<std::uint32_t, 8> required;
    };

    std::uint64_t sinkOffset = 0;
    std::uint32_t sinkCount = 0;
    std::uint32_t consumedPhysicalUseCount = 0;
    // Derived once from the frozen publication sinks and PairingKey domain.
    llvm::SmallVector<OperandIngressQuery, 2> operandIngressQueries;
    llvm::SmallVector<::loom::mapping::SpatialPeOperandQualifiedPairingKey, 4>
        operandPairings;
  };

  struct TransferBinding final {
    ::dataflow::CanonicalGraphProducerEndpointRef producer;
    std::uint64_t sinkOffset = 0;
    std::uint32_t sinkCount = 0;
    std::uint64_t physicalUseOffset = 0;
    std::uint32_t physicalUseCount = 0;
    std::uint64_t traversalNodeOffset = 0;
    std::uint32_t traversalNodeCount = 0;
    std::uint32_t traversalTerminalCount = 0;
    std::uint32_t consumedPhysicalUseCount = 0;
    std::uint64_t publicationOffset = 0;
    std::uint32_t publicationCount = 0;
    std::optional<std::uint64_t> semanticActorOrdinal;
    bool discard = false;
  };

  enum class TraversalNodeKind : std::uint8_t {
    PhysicalAction,
    BufferedStorage,
    RegisterStorageWrite,
    RegisterStorageRead,
  };

  struct TraversalNodeBinding final {
    TraversalNodeKind kind = TraversalNodeKind::PhysicalAction;
    std::uint64_t physicalUseOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t storageOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t physicalTagOrdinal = invalidCgraTransportOrdinal;
    std::uint64_t targetTraversalOffset = 0;
    std::uint32_t targetTraversalCount = 0;
    std::uint64_t successorOffset = 0;
    std::uint32_t successorCount = 0;
    std::uint32_t predecessorCount = 0;
    bool terminal = false;
    std::vector<std::uint32_t> descendantSinks;
    std::vector<std::uint32_t> terminalSinks;
    std::vector<std::uint64_t> downstreamStorageNodes;
    std::vector<std::uint32_t> unbufferedDescendantSinks;
  };

  struct StorageBinding final {
    CgraTraversalStorageKind kind = CgraTraversalStorageKind::None;
    std::uint32_t capacity = 0;
    ::fabric::FifoQueueDiscipline queueDiscipline =
        ::fabric::FifoQueueDiscipline::StrictFifo;
    /// Channels a per-tag virtual channel pool guarantees one slot each.
    std::uint32_t reservedChannels = 0;
    bool independentReadWriteServices = false;
    std::uint64_t enqueueAction = invalidCgraTransportOrdinal;
    std::uint64_t dequeueAction = invalidCgraTransportOrdinal;
    std::uint64_t simultaneousAction = invalidCgraTransportOrdinal;
    std::uint64_t offerAdvanceAction = invalidCgraTransportOrdinal;
    std::vector<std::uint64_t> upstreamStorageOrdinals;
  };

  struct OperandQueueUnitBinding final {
    ::loom::fabric::FabricPeOccurrenceRef pe;
    std::uint32_t allocationUnit = 0;
    std::uint32_t capacity = 0;
  };

  struct OperandBufferBinding final {
    ::loom::fabric::FabricPeOccurrenceRef pe;
    const ::fabric::TemporalOperandBufferContract &contract;
    std::vector<std::uint64_t> runtimeQueues;
    std::vector<std::uint64_t> runtimeUnits;
  };

  struct OperandQueueBinding final {
    struct Consumer final {
      ChannelOrdinal channel = 0;
      std::uint64_t semanticActorOrdinal = 0;
      unsigned inputOrdinal = 0;
    };

    ::fabric::LogicalOperandQueueKey queue;
    ::loom::fabric::FabricFuOccurrenceRef fu;
    std::uint64_t bufferBinding = invalidCgraTransportOrdinal;
    std::uint32_t contractQueue = 0;
    std::uint64_t unitBinding = invalidCgraTransportOrdinal;
    std::vector<Consumer> consumers;
  };

  std::vector<TransferBinding> bindings;
  std::vector<SinkBinding> sinks;
  std::vector<PublicationBinding> publications;
  std::vector<std::uint32_t> publicationSinks;
  std::vector<std::uint64_t> physicalUses;
  std::vector<TraversalNodeBinding> traversalNodes;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversalTargets;
  std::vector<std::uint64_t> traversalSuccessors;
  std::vector<StorageBinding> storages;
  std::vector<OperandBufferBinding> operandBuffers;
  std::vector<OperandQueueUnitBinding> operandQueueUnits;
  std::vector<OperandQueueBinding> operandQueues;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorSourceBindings;
  std::vector<llvm::SmallVector<std::uint64_t, 2>> actorSourceBindingOrdinals;
  llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorInputQueueBindings;
  std::vector<std::uint32_t> tagVirtualChannelKeys;
};

llvm::Expected<CgraTransportGraph> freezeCgraTransportGraph(
    const CgraFrozenExecutionPlan &plan,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::GraphRef graph, const PreparedGraphExecution &execution);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTGRAPH_H
