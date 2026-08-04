#ifndef LOOM_LIB_SIMULATOR_CGRACOMPUTERUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRACOMPUTERUNTIME_H

#include "CGRAExecutionPlan.h"
#include "DFGSimulatorInternal.h"

#include "Simulator/CGRA/EventQueue.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::sim::detail {

enum class CgraComputeActorLifecycleKind : std::uint8_t {
  Committed,
  Retired,
};

struct CgraComputeActorLifecycleEvent final {
  CgraComputeActorLifecycleKind kind = CgraComputeActorLifecycleKind::Committed;
  std::uint64_t actorPlanOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t transitionCaseOrdinal = 0;
  SpatialEventCoordinate coordinate;
};

struct CgraComputeActorEmission final {
  std::uint64_t actorPlanOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t transitionCaseOrdinal = 0;
  unsigned resultOrdinal = 0;
  Token token;
};

/// Internal handoff from compute/resource execution to transport retirement.
/// It is a transient dense projection, not a trace event or persistent ref.
struct CgraTransitionPhysicalCompletion final {
  std::uint64_t actorPlanOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t transitionCaseOrdinal = 0;
};

struct CgraComputeLifecycleFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraPhysicalLifecycleEvent> physicalEvents;
  std::vector<CgraComputeActorLifecycleEvent> actorEvents;
  std::vector<CgraComputeActorEmission> actorEmissions;
  std::vector<CgraTransitionPhysicalCompletion> physicalCompletions;
};

/// Execution-local compute/resource state for one mapped graph activation.
/// Canonical actor semantics remain in PreparedGraphExecution, while physical
/// timing and arbitration remain in CgraPhysicalActionRuntime.
class CgraComputeRuntime final {
public:
  static llvm::Expected<CgraComputeRuntime>
  create(const CgraFrozenExecutionPlan &plan,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         ::dataflow::GraphRef graph, const PreparedGraphExecution &execution,
         SimulatorState &state, CgraPhysicalActionRuntime &physical);

  llvm::Error start(SpatialEventCoordinate coordinate);

  llvm::Error
  acceptReadyCandidates(SpatialEventCoordinate coordinate,
                        const llvm::SmallBitVector &semanticCandidates);

  llvm::Expected<CgraComputeLifecycleFrame>
  acceptPhysicalEvents(const CgraPhysicalLifecycleFrame &physicalFrame);

  llvm::Error retireActor(std::uint64_t actorPlanOrdinal,
                          std::uint64_t occurrenceOrdinal,
                          SpatialEventCoordinate coordinate);

  llvm::Expected<std::optional<CgraComputeLifecycleFrame>> advance();

  std::optional<SpatialEventCoordinate> nextCoordinate() const;

  bool hasPendingEvents() const;
  bool hasActiveActors() const { return activeActorCount_ != 0; }

private:
  struct ActorBinding final {
    std::uint64_t actorPlanOrdinal = 0;
    const ActorExecutionPlan *semantic = nullptr;
    std::uint64_t transitionIndexOffset = 0;
    std::uint32_t transitionCount = 0;
    std::uint64_t nextOccurrenceOrdinal = 0;
    bool commitPending = false;
    bool retirementPending = false;
    std::uint64_t activeOccurrenceOrdinal = 0;
  };

  struct Firing final {
    bool active = false;
    std::uint64_t bindingOrdinal = 0;
    std::uint64_t actorOccurrenceOrdinal = 0;
    std::uint32_t transitionCaseOrdinal = 0;
    bool commitScheduled = false;
    bool committed = false;
    std::uint32_t actionCount = 0;
    std::uint32_t permittedCount = 0;
    std::uint32_t retiredCount = 0;
  };

  struct FiringActionIndex final {
    std::uint64_t firingSlot = 0;
  };

  CgraComputeRuntime(const CgraFrozenExecutionPlan &plan, SimulatorState &state,
                     std::vector<ActorBinding> bindings,
                     std::vector<std::uint64_t> transitionByCase,
                     std::vector<std::uint64_t> bindingBySemanticActor,
                     std::vector<std::uint64_t> bindingByActorPlan,
                     CgraPhysicalActionRuntime &physical);

  llvm::Error scheduleReady(SpatialEventCoordinate coordinate);
  llvm::Expected<std::uint64_t>
  allocateFiring(std::uint64_t bindingOrdinal,
                 const CgraComputeTransitionPlan &transition);
  llvm::Error
  processPhysicalEvent(const CgraPhysicalLifecycleEvent &event,
                       CgraComputeLifecycleFrame &frame,
                       llvm::SmallVectorImpl<std::uint64_t> &affectedFirings);
  llvm::Error processActorCommit(std::uint64_t firingSlot,
                                 CgraComputeLifecycleFrame &frame);
  llvm::Error maybeScheduleCommit(std::uint64_t firingSlot,
                                  const SpatialEventCoordinate &coordinate);
  void maybeComplete(std::uint64_t firingSlot,
                     CgraComputeLifecycleFrame &frame);
  void releaseFiring(std::uint64_t firingSlot);

  const CgraFrozenExecutionPlan *plan_ = nullptr;
  SimulatorState *state_ = nullptr;
  std::vector<ActorBinding> bindings_;
  std::vector<std::uint64_t> transitionByCase_;
  std::vector<std::uint64_t> bindingBySemanticActor_;
  std::vector<std::uint64_t> bindingByActorPlan_;
  CgraPhysicalActionRuntime *physical_ = nullptr;
  CgraEventQueue requestedEvents_;
  CgraEventQueue actorCommitEvents_;
  llvm::SmallBitVector readyCandidates_;
  std::vector<std::uint64_t> nextActionOccurrence_;
  std::vector<Firing> firings_;
  std::vector<std::uint64_t> freeFiringSlots_;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, FiringActionIndex>
      actionToFiring_;
  std::uint64_t activeActorCount_ = 0;
  bool started_ = false;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRACOMPUTERUNTIME_H
