#ifndef LOOM_LIB_SIMULATOR_CGRAMEMORYRUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRAMEMORYRUNTIME_H

#include "CgraComputeRuntime.h"

#include "Simulator/CgraExternalMemoryProvider.h"
#include "Simulator/SpatialTrace.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::sim::detail {

struct CgraMemoryLifecycleFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraPhysicalLifecycleEvent> physicalEvents;
  std::vector<CgraActorLifecycleEvent> actorEvents;
  std::vector<CgraActorEmission> actorEmissions;
  std::vector<CgraActorPhysicalCompletion> physicalCompletions;
  std::vector<MemoryLinearizedTraceEvent> memoryLinearizations;
};

/// Dynamic execution of mapped Dataflow load/store actors. Canonical Dataflow
/// remains the functional authority and CgraMemoryPlan remains the selected
/// physical authority; this class only coordinates their issue,
/// linearization, and retirement moments.
class CgraMemoryRuntime final {
public:
  static llvm::Expected<CgraMemoryRuntime>
  create(const CgraFrozenExecutionPlan &plan,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         ::dataflow::RootedGraphLaunchRef launch, ::dataflow::GraphRef graph,
         const PreparedGraphExecution &execution, SimulatorState &state,
         CgraPhysicalActionRuntime &physical,
         CgraExternalMemoryProvider *externalMemoryProvider);

  llvm::Error start(SpatialEventCoordinate coordinate);

  llvm::Error
  acceptReadyCandidates(SpatialEventCoordinate coordinate,
                        const llvm::SmallBitVector &semanticCandidates);

  llvm::Expected<CgraMemoryLifecycleFrame>
  acceptPhysicalEvents(const CgraPhysicalLifecycleFrame &physicalFrame);

  llvm::Expected<CgraPhysicalTraceBinding>
  physicalTraceBinding(const CgraPhysicalLifecycleEvent &event) const;

  llvm::Error retireActor(std::uint64_t semanticActorOrdinal,
                          std::uint64_t occurrenceOrdinal,
                          SpatialEventCoordinate coordinate, bool reschedule);

  llvm::Expected<std::optional<CgraMemoryLifecycleFrame>> advance();

  std::optional<SpatialEventCoordinate> nextCoordinate() const {
    return requestedEvents_.nextCoordinate();
  }

  bool hasPendingEvents() const {
    return !requestedEvents_.empty() || activeActorCount_ != 0;
  }
  bool hasActiveActors() const { return activeActorCount_ != 0; }
  std::uint64_t activeActorCount() const { return activeActorCount_; }
  bool ownsActor(std::uint64_t semanticActorOrdinal) const;

private:
  struct ResultBinding final {
    ::dataflow::semantics::ServiceValueRole role =
        ::dataflow::semantics::ServiceValueRole::Completion;
    unsigned resultOrdinal = 0;
    std::optional<std::uint64_t> assemblyOrdinal;
  };

  struct ActorBinding final {
    std::uint64_t semanticActorOrdinal = 0;
    const ActorExecutionPlan *semantic = nullptr;
    const CgraMemoryActorPlan *physical = nullptr;
    const CgraMemoryRootedUsePlan *rootedUse = nullptr;
    std::vector<ResultBinding> results;
    std::uint64_t nextOccurrenceOrdinal = 0;
    bool retirementPending = false;
    std::uint64_t activeOccurrenceOrdinal = 0;
  };

  struct Firing final {
    bool active = false;
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t actorOccurrenceOrdinal = 0;
    std::optional<ReadyPlainMemoryAction> ready;
    std::optional<Token> storeData;
    std::uint32_t activeChildCount = 0;
    std::uint32_t permittedChildCount = 0;
    std::uint32_t retiredChildCount = 0;
    bool operationPermitted = false;
    bool operationRetired = false;
    bool issueCommitted = false;
    bool linearized = false;
  };

  struct ActionIndex final {
    std::uint64_t firingSlot = 0;
    std::uint64_t localActionOrdinal = 0;
    bool operation = false;
  };

  CgraMemoryRuntime(const CgraFrozenExecutionPlan &plan, SimulatorState &state,
                    std::vector<ActorBinding> bindings,
                    std::vector<std::uint64_t> bindingBySemanticActor,
                    CgraPhysicalActionRuntime &physical,
                    CgraExternalMemoryProvider *externalMemoryProvider)
      : plan_(&plan), state_(&state), bindings_(std::move(bindings)),
        bindingBySemanticActor_(std::move(bindingBySemanticActor)),
        physical_(&physical), externalMemoryProvider_(externalMemoryProvider),
        nextActionOccurrence_(plan.physicalUseTimings.size(), 0) {}

  llvm::Error scheduleReady(SpatialEventCoordinate coordinate);
  llvm::Expected<std::uint64_t> allocateFiring(std::uint64_t bindingOrdinal,
                                               ReadyPlainMemoryAction ready,
                                               std::optional<Token> storeData);
  llvm::Expected<CgraPhysicalLifecycleEvent>
  requestAction(std::uint64_t firingSlot, std::uint64_t actionOrdinal,
                std::uint64_t localActionOrdinal, bool operation,
                const SpatialEventCoordinate &coordinate);
  llvm::Error processPhysicalEvent(const CgraPhysicalLifecycleEvent &event,
                                   CgraMemoryLifecycleFrame &frame);
  llvm::Error commitIssue(std::uint64_t firingSlot,
                          const SpatialEventCoordinate &coordinate,
                          CgraMemoryLifecycleFrame &frame);
  llvm::Error linearize(std::uint64_t firingSlot,
                        CgraMemoryLifecycleFrame &frame);
  void maybeComplete(std::uint64_t firingSlot, CgraMemoryLifecycleFrame &frame);
  void releaseFiring(std::uint64_t firingSlot);

  const CgraFrozenExecutionPlan *plan_ = nullptr;
  SimulatorState *state_ = nullptr;
  std::vector<ActorBinding> bindings_;
  std::vector<std::uint64_t> bindingBySemanticActor_;
  CgraPhysicalActionRuntime *physical_ = nullptr;
  CgraExternalMemoryProvider *externalMemoryProvider_ = nullptr;
  CgraEventQueue requestedEvents_{"CGRA memory request"};
  std::vector<std::uint64_t> nextActionOccurrence_;
  std::vector<Firing> firings_;
  std::vector<std::uint64_t> freeFiringSlots_;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, ActionIndex>
      actionToFiring_;
  std::uint64_t activeActorCount_ = 0;
  bool started_ = false;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAMEMORYRUNTIME_H
