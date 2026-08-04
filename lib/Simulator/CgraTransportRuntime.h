#ifndef LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H

#include "CgraComputeRuntime.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::sim::detail {

struct CgraTokenPublication final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::uint64_t occurrenceOrdinal = 0;
  Token token;
};

struct CgraTransportCompletion final {
  std::uint64_t actorPlanOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
};

struct CgraTransportFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraPhysicalLifecycleEvent> physicalEvents;
  std::vector<CgraTokenPublication> publications;
  std::vector<CgraTransportCompletion> completions;
  std::vector<std::uint64_t> blockedTransfers;
};

/// Execution-local token transport for one mapped graph activation. It binds
/// exact Dataflow endpoints to dense DFG channel slots once; dynamic events
/// never search MLIR users or persistent-reference bytes.
class CgraTransportRuntime final {
public:
  static llvm::Expected<CgraTransportRuntime>
  create(const CgraFrozenExecutionPlan &plan,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         ::dataflow::GraphRef graph, const PreparedGraphExecution &execution,
         SimulatorState &state, CgraPhysicalActionRuntime &physical);

  llvm::Error acceptActorEmissions(
      const SpatialEventCoordinate &coordinate,
      llvm::MutableArrayRef<CgraComputeActorEmission> emissions);

  llvm::Error acceptGraphIngressEmissions(
      const SpatialEventCoordinate &coordinate,
      llvm::MutableArrayRef<GraphIngressEmission> emissions);

  llvm::Expected<std::vector<CgraTransportCompletion>>
  acceptPhysicalEvents(const CgraPhysicalLifecycleFrame &physicalFrame);

  llvm::Expected<std::optional<CgraTransportFrame>> advance();

  llvm::Error retryBlocked(const SpatialEventCoordinate &coordinate);

  std::optional<SpatialEventCoordinate> nextCoordinate() const;

  bool hasPendingEvents() const {
    return !arrivalEvents_.empty() || !events_.empty() ||
           !requestedEvents_.empty() || activeTransferCount_ != 0;
  }
  bool hasBlockedTransfers() const { return blocked_.any(); }

private:
  enum class SinkKind : std::uint8_t { Channel, Observation };

  struct SinkBinding final {
    SinkKind kind = SinkKind::Channel;
    ChannelOrdinal channel = 0;
    mlir::Value observation;
    std::uint64_t physicalUseOffset = 0;
    std::uint32_t physicalUseCount = 0;
  };

  struct TransferBinding final {
    ::dataflow::CanonicalGraphProducerEndpointRef producer;
    std::uint64_t sinkOffset = 0;
    std::uint32_t sinkCount = 0;
    std::uint64_t physicalUseOffset = 0;
    std::uint32_t physicalUseCount = 0;
    std::uint32_t consumedPhysicalUseCount = 0;
    std::optional<std::uint64_t> actorPlanOrdinal;
    bool requiresTraversalTransport = false;
    bool active = false;
  };

  struct InFlight final {
    bool active = false;
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    Token token;
    bool arrivalScheduled = false;
    bool publicationScheduled = false;
    bool published = false;
    bool consumedRequested = false;
    std::uint32_t producedPermitted = 0;
    std::uint32_t producedRetired = 0;
    std::uint32_t consumedPermitted = 0;
    std::uint32_t consumedRetired = 0;
  };

  enum class ActionStage : std::uint8_t { Produced, Consumed };

  enum class ActionLifecycleState : std::uint8_t {
    Requested,
    Granted,
    Permitted,
    Retired,
  };

  struct ActionOwner final {
    std::uint64_t transferSlot = 0;
    ActionStage stage = ActionStage::Produced;
    ActionLifecycleState state = ActionLifecycleState::Requested;
  };

  struct PendingTransfer final {
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    Token *token = nullptr;
  };

  struct PendingActionTransfer final {
    std::uint64_t transferSlot = 0;
    std::uint64_t bindingOrdinal = 0;
  };

  CgraTransportRuntime(
      const CgraFrozenExecutionPlan &plan, SimulatorState &state,
      CgraPhysicalActionRuntime &physical,
      std::vector<TransferBinding> bindings, std::vector<SinkBinding> sinks,
      std::vector<std::uint64_t> physicalUses,
      llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
          actorSourceBindings,
      llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings);

  std::uint64_t allocate(std::uint64_t bindingOrdinal,
                         std::uint64_t occurrenceOrdinal, Token token);
  llvm::Error acceptTransfers(const SpatialEventCoordinate &coordinate,
                              llvm::ArrayRef<PendingTransfer> transfers);
  llvm::Expected<std::vector<CgraPhysicalLifecycleEvent>>
  requestActions(llvm::ArrayRef<PendingActionTransfer> transfers,
                 ActionStage stage, const SpatialEventCoordinate &coordinate);
  llvm::Error scheduleArrival(std::uint64_t slot,
                              const SpatialEventCoordinate &coordinate);
  llvm::Error schedulePublication(std::uint64_t slot,
                                  const SpatialEventCoordinate &coordinate);
  std::optional<CgraTransportCompletion> maybeRelease(std::uint64_t slot);
  void scheduleAt(std::uint64_t slot,
                  const SpatialEventCoordinate &publicationCoordinate);
  bool canPublish(const TransferBinding &binding) const;
  void publish(std::uint64_t slot, CgraTransportFrame &frame);
  std::optional<CgraTransportCompletion> release(std::uint64_t slot);

  const CgraFrozenExecutionPlan *plan_ = nullptr;
  SimulatorState *state_ = nullptr;
  CgraPhysicalActionRuntime *physical_ = nullptr;
  std::vector<TransferBinding> bindings_;
  std::vector<SinkBinding> sinks_;
  std::vector<std::uint64_t> physicalUses_;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorSourceBindings_;
  llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings_;
  CgraEventQueue events_;
  CgraEventQueue arrivalEvents_;
  CgraEventQueue requestedEvents_;
  std::vector<InFlight> inFlight_;
  std::vector<std::uint64_t> freeSlots_;
  llvm::SmallBitVector blocked_;
  std::vector<std::uint64_t> nextActionOccurrence_;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, ActionOwner>
      actionOwners_;
  std::uint64_t activeTransferCount_ = 0;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H
