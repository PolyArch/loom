#ifndef LOOM_LIB_SIMULATOR_CGRACOMPUTERUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRACOMPUTERUNTIME_H

#include "CGRAExecutionPlan.h"
#include "CgraPhysicalTraceProjection.h"
#include "DFGSimulatorInternal.h"

#include "Simulator/CGRA/EventQueue.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace loom::sim::detail {

enum class CgraActorLifecycleKind : std::uint8_t {
  Committed,
  Retired,
};

struct CgraActorLifecycleEvent final {
  CgraActorLifecycleKind kind = CgraActorLifecycleKind::Committed;
  std::uint64_t semanticActorOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t transitionCaseOrdinal = 0;
  std::uint32_t expectedTransferCount = 0;
  SpatialEventCoordinate coordinate;
};

struct CgraActorEmission final {
  std::uint64_t semanticActorOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t transitionCaseOrdinal = 0;
  unsigned resultOrdinal = 0;
  Token token;
};

/// Internal handoff from compute/resource execution to transport retirement.
/// It is a transient dense projection, not a trace event or persistent ref.
struct CgraActorPhysicalCompletion final {
  std::uint64_t semanticActorOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t transitionCaseOrdinal = 0;
};

struct CgraComputeLifecycleFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraPhysicalLifecycleEvent> physicalEvents;
  std::vector<CgraActorLifecycleEvent> actorEvents;
  std::vector<CgraActorEmission> actorEmissions;
  std::vector<CgraActorPhysicalCompletion> physicalCompletions;
};

/// Projects one active temporal-PE context-evaluation candidate onto its next
/// Fabric-owned round-robin service slot. This is the execution oracle shared
/// by the runtime and its semantic tests; actor readiness and transition
/// commit remain separate concerns.
llvm::Expected<SpatialEventCoordinate> projectCgraTemporalDispatchCoordinate(
    const CgraTemporalDispatchDomainPlan &domain,
    std::uint32_t candidatePosition, const SpatialEventCoordinate &coordinate);

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

  llvm::Expected<CgraPhysicalTraceBinding>
  physicalTraceBinding(const CgraPhysicalLifecycleEvent &event) const;

  llvm::Error retireActor(std::uint64_t semanticActorOrdinal,
                          std::uint64_t occurrenceOrdinal,
                          SpatialEventCoordinate coordinate, bool reschedule);

  llvm::Error satisfyCausalRelease(std::uint64_t semanticActorOrdinal,
                                   std::uint64_t occurrenceOrdinal,
                                   std::uint32_t transitionCaseOrdinal,
                                   SpatialEventCoordinate coordinate);

  llvm::Expected<std::optional<CgraComputeLifecycleFrame>> advance();

  std::optional<SpatialEventCoordinate> nextCoordinate() const;

  bool hasPendingEvents() const;
  bool hasActiveActors() const { return activeActorCount_ != 0; }
  std::uint64_t activeActorCount() const { return activeActorCount_; }
  bool ownsActor(std::uint64_t semanticActorOrdinal) const {
    return semanticActorOrdinal < bindingBySemanticActor_.size() &&
           bindingBySemanticActor_[semanticActorOrdinal] !=
               std::numeric_limits<std::uint64_t>::max();
  }

private:
  struct ActorBinding final {
    std::uint64_t semanticActorOrdinal = 0;
    ::dataflow::ActorRef actor;
    const ActorExecutionPlan *semantic = nullptr;
    std::uint64_t transitionIndexOffset = 0;
    std::uint32_t transitionCount = 0;
    std::optional<std::uint64_t> temporalDispatchDomain;
    std::uint32_t temporalDispatchPosition = 0;
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
    std::uint32_t completedCount = 0;
    std::uint32_t retiredCount = 0;
    bool completionReported = false;
  };

  struct FiringActionIndex final {
    std::uint64_t firingSlot = 0;
    std::uint64_t localActionOrdinal = 0;
  };

  CgraComputeRuntime(const CgraFrozenExecutionPlan &plan, SimulatorState &state,
                     std::vector<ActorBinding> bindings,
                     std::vector<std::uint64_t> transitionByCase,
                     std::vector<std::uint64_t> bindingBySemanticActor,
                     CgraPhysicalActionRuntime &physical);

  llvm::Error scheduleReady(SpatialEventCoordinate coordinate);
  llvm::Expected<SpatialEventCoordinate>
  dispatchCoordinate(const ActorBinding &binding,
                     const SpatialEventCoordinate &coordinate) const;
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
  CgraPhysicalActionRuntime *physical_ = nullptr;
  CgraEventQueue requestedEvents_{"CGRA compute request"};
  CgraEventQueue actorCommitEvents_{"CGRA actor commit"};
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
