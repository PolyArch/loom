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

struct CgraTransportFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraTokenPublication> publications;
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
         SimulatorState &state);

  llvm::Error acceptActorEmissions(
      const SpatialEventCoordinate &coordinate,
      llvm::MutableArrayRef<CgraComputeActorEmission> emissions);

  llvm::Error acceptGraphIngressEmissions(
      const SpatialEventCoordinate &coordinate,
      llvm::MutableArrayRef<GraphIngressEmission> emissions);

  llvm::Expected<std::optional<CgraTransportFrame>> advance();

  llvm::Error retryBlocked(const SpatialEventCoordinate &coordinate);

  bool hasPendingEvents() const { return !events_.empty(); }
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
    bool requiresPhysicalTransport = false;
    bool active = false;
  };

  struct InFlight final {
    bool active = false;
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    Token token;
  };

  CgraTransportRuntime(
      SimulatorState &state, std::vector<TransferBinding> bindings,
      std::vector<SinkBinding> sinks, std::vector<std::uint64_t> physicalUses,
      llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
          actorSourceBindings,
      llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings);

  std::uint64_t allocate(std::uint64_t bindingOrdinal,
                         std::uint64_t occurrenceOrdinal, Token token);
  void scheduleAt(std::uint64_t slot,
                  const SpatialEventCoordinate &publicationCoordinate);
  bool canPublish(const TransferBinding &binding) const;
  void publish(std::uint64_t slot, CgraTransportFrame &frame);
  void release(std::uint64_t slot);

  SimulatorState *state_ = nullptr;
  std::vector<TransferBinding> bindings_;
  std::vector<SinkBinding> sinks_;
  std::vector<std::uint64_t> physicalUses_;
  llvm::DenseMap<std::pair<std::uint64_t, unsigned>, std::uint64_t>
      actorSourceBindings_;
  llvm::DenseMap<unsigned, std::uint64_t> ingressSourceBindings_;
  CgraEventQueue events_;
  std::vector<InFlight> inFlight_;
  std::vector<std::uint64_t> freeSlots_;
  llvm::SmallBitVector blocked_;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRATRANSPORTRUNTIME_H
