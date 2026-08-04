#ifndef LOOM_LIB_SIMULATOR_CGRAGRAPHACTIVATIONRUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRAGRAPHACTIVATIONRUNTIME_H

#include "CgraMemoryRuntime.h"
#include "CgraTransportRuntime.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace loom::sim::detail {

struct CgraGraphActivationFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraPhysicalLifecycleEvent> physicalEvents;
  std::vector<CgraActorLifecycleEvent> actorEvents;
  std::vector<CgraTokenPublication> publications;
};

/// One execution-local coordinator for a mapped graph activation. It alone
/// advances the shared physical calendar and distributes each frame to the
/// compute and transport clients before exposing the next coordinate.
class CgraGraphActivationRuntime final {
public:
  static llvm::Expected<CgraGraphActivationRuntime>
  create(const CgraFrozenExecutionPlan &plan,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         ::dataflow::RootedGraphLaunchRef launch, ::dataflow::GraphRef graph,
         const PreparedGraphExecution &execution, SimulatorState &state);

  llvm::Error start(SpatialEventCoordinate coordinate,
                    llvm::MutableArrayRef<GraphIngressEmission> ingress);

  llvm::Expected<std::optional<CgraGraphActivationFrame>> advance();

  std::optional<SpatialEventCoordinate> nextCoordinate() const;
  bool hasPendingEvents() const;

private:
  struct ActorFiring final {
    bool active = false;
    std::uint64_t semanticActorOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    std::uint32_t transitionCaseOrdinal = 0;
    std::uint32_t expectedTransfers = 0;
    std::uint32_t completedTransfers = 0;
    bool physicalComplete = false;
  };

  CgraGraphActivationRuntime(
      SimulatorState &state,
      std::unique_ptr<CgraPhysicalActionRuntime> physical,
      std::unique_ptr<CgraComputeRuntime> compute,
      std::unique_ptr<CgraMemoryRuntime> memory,
      std::unique_ptr<CgraTransportRuntime> transport)
      : state_(&state), physical_(std::move(physical)),
        compute_(std::move(compute)), memory_(std::move(memory)),
        transport_(std::move(transport)) {}

  llvm::Error consumeComputeFrame(CgraComputeLifecycleFrame frame,
                                  CgraGraphActivationFrame &result);
  llvm::Error consumeMemoryFrame(CgraMemoryLifecycleFrame frame,
                                 CgraGraphActivationFrame &result);
  llvm::Error consumeTransportFrame(CgraTransportFrame frame,
                                    CgraGraphActivationFrame &result);
  llvm::Error consumeTransportCompletions(
      llvm::ArrayRef<CgraTransportCompletion> completions,
      const SpatialEventCoordinate &coordinate,
      CgraGraphActivationFrame &result);
  llvm::Error
  markPhysicalCompletion(const CgraActorPhysicalCompletion &completion,
                         const SpatialEventCoordinate &coordinate,
                         CgraGraphActivationFrame &result);
  llvm::Error maybeRetire(std::uint64_t firingSlot,
                          const SpatialEventCoordinate &coordinate,
                          CgraGraphActivationFrame &result);
  llvm::Expected<std::uint64_t>
  addCommittedFiring(const CgraActorLifecycleEvent &event);
  void releaseFiring(std::uint64_t firingSlot);
  llvm::Error
  schedulePublishedCandidates(const SpatialEventCoordinate &coordinate);

  SimulatorState *state_ = nullptr;
  std::unique_ptr<CgraPhysicalActionRuntime> physical_;
  std::unique_ptr<CgraComputeRuntime> compute_;
  std::unique_ptr<CgraMemoryRuntime> memory_;
  std::unique_ptr<CgraTransportRuntime> transport_;
  std::vector<ActorFiring> firings_;
  std::vector<std::uint64_t> freeFiringSlots_;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t>
      firingByOccurrence_;
  bool started_ = false;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAGRAPHACTIVATIONRUNTIME_H
