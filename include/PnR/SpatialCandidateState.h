#ifndef LOOM_PNR_SPATIALCANDIDATESTATE_H
#define LOOM_PNR_SPATIALCANDIDATESTATE_H

#include "PnR/HandshakeCandidateState.h"
#include "PnR/RouteTreeState.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialProgressState.h"
#include "PnR/SpatialRecurrenceTiming.h"
#include "PnR/SpatialRouteResourceState.h"
#include "PnR/SpatialTagAssignment.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::pnr {

namespace detail {
class SpatialMemoryConstraintScratch;
class SpatialRouteConstraintScratch;
} // namespace detail

struct SpatialComputeBindingSelection final {
  PnrIndex placement = getInvalidPnrIndex();
  PnrIndex instructionContext = getInvalidPnrIndex();
};

struct SpatialMemoryBindingSelection final {
  PnrIndex placement = getInvalidPnrIndex();
};

struct SpatialLogicalMemoryBindingSelection final {
  PnrIndex target = getInvalidPnrIndex();
  std::uint64_t physicalOffsetBytes = 0;
};

struct SpatialCandidateInitialization final {
  llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings;
  llvm::ArrayRef<SpatialMemoryBindingSelection> memoryBindings;
  llvm::ArrayRef<PnrIndex> portAttachments;
  llvm::ArrayRef<PnrIndex> graphBoundaryAttachments;
  llvm::ArrayRef<PnrIndex> memoryOperationPlans;
  llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> logicalMemoryBindings;
  llvm::ArrayRef<PnrIndex> memoryUseDispatches;
  llvm::ArrayRef<PnrIndex> memoryExposureSelections;
  llvm::ArrayRef<PnrIndex> registerFifoTransfers;
};

/// Cold reconstruction of every route-derived Mapping fact for the current
/// RouteTrees. It is independent of commit-time caches and is therefore valid
/// while an enclosing move still owns provisional routes.
struct SpatialCandidateRouteProjection final {
  std::uint64_t unroutedObligationCount = 0;
  std::uint64_t routeCapacityOveruse = 0;
  std::uint64_t tagResidentCapacityOveruse = 0;
  std::uint64_t tagUnassignedCount = 0;
  std::uint64_t tagConflictCount = 0;
  std::uint64_t hardProgressViolation = 0;
  std::uint64_t totalSelectedTraversalClaim = 0;
  std::uint64_t routeReleaseLatencyCycles = 0;
  std::uint64_t routeMinimumInitiationIntervalCycles = 1;
  SpatialRecurrenceTimingProjection recurrenceTiming;
  std::uint64_t transportBitCycleDemand = 0;
  std::uint64_t worstRouteArrivalDelayQuanta = 0;
  std::uint64_t totalRouteNegativeSlackQuanta = 0;
  bool routeTerminalsCompatible = false;
  bool selectedHandshakeAcyclic = false;
};

class SpatialCandidateState;
class SpatialCandidateScratch;
class SpatialActionDomainScratch;
class SpatialActionExecutorScratch;
class SpatialMoveTransaction;

using SpatialCandidateStateHandle = std::shared_ptr<SpatialCandidateState>;

class SpatialCandidateScratch final {
public:
  SpatialCandidateScratch();
  SpatialCandidateScratch(const SpatialCandidateScratch &) = delete;
  SpatialCandidateScratch &operator=(const SpatialCandidateScratch &) = delete;
  SpatialCandidateScratch(SpatialCandidateScratch &&) = delete;
  SpatialCandidateScratch &operator=(SpatialCandidateScratch &&) = delete;
  ~SpatialCandidateScratch();

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);
  std::size_t retainedStorageBytes() const;
  HandshakeProjectionStatistics handshakeProjectionStatistics() const {
    return handshakeProjectionScratch_.statistics();
  }

private:
  enum class DecisionKind : std::uint8_t {
    ComputeBinding,
    MemoryBinding,
    PortAttachment,
    GraphBoundaryAttachment,
    MemoryOperationPlan,
    LogicalMemoryBinding,
    MemoryUseDispatch,
    MemoryExposure,
    RegisterFifoTransfer,
  };

  struct DecisionDelta final {
    DecisionKind kind = DecisionKind::ComputeBinding;
    PnrIndex index = 0;
    PnrIndex oldValue0 = 0;
    PnrIndex oldValue1 = 0;
    PnrIndex oldValue2 = 0;
    std::uint64_t oldWideValue = 0;
  };

  struct ProgressTraversalDelta final {
    PnrIndex logicalNet = getInvalidPnrIndex();
    PnrIndex traversal = getInvalidPnrIndex();
    PnrIndex removed = 0;
    PnrIndex added = 0;
  };

  struct ProgressDependencyDelta final {
    PnrIndex logicalNet = getInvalidPnrIndex();
    std::uint64_t oldCount = 0;
  };

  void beginTransaction();
  void advanceProgressRouteDeltaEpoch();
  void resetTransaction();

  std::vector<std::unique_ptr<RouteTreeTransactionScratch>> routeScratch_;
  std::vector<std::optional<RouteTreeTransaction>> routeTransactions_;
  SpatialTagAssignmentScratch tagScratch_;
  HandshakeCandidateScratch handshakeScratch_;
  HandshakeProjectionScratch handshakeProjectionScratch_;
  std::optional<HandshakeCandidateTransaction> handshakeTransaction_;

  std::vector<std::uint64_t> computeJournalMarks_;
  std::vector<std::uint64_t> memoryJournalMarks_;
  std::vector<std::uint64_t> portJournalMarks_;
  std::vector<std::uint64_t> boundaryJournalMarks_;
  std::vector<std::uint64_t> memoryPlanJournalMarks_;
  std::vector<std::uint64_t> logicalMemoryJournalMarks_;
  std::vector<std::uint64_t> memoryDispatchJournalMarks_;
  std::vector<std::uint64_t> memoryExposureJournalMarks_;
  std::vector<std::uint64_t> registerFifoTransferJournalMarks_;
  std::vector<DecisionDelta> decisionDeltas_;
  std::uint64_t decisionEpoch_ = 0;

  std::vector<std::uint64_t> affectedComputeMarks_;
  std::vector<std::uint64_t> affectedMemoryMarks_;
  std::vector<std::uint64_t> affectedPortMarks_;
  std::vector<std::uint64_t> affectedBoundaryMarks_;
  std::vector<std::uint64_t> affectedMemoryPlanMarks_;
  std::vector<std::uint64_t> affectedLogicalMemoryMarks_;
  std::vector<std::uint64_t> affectedMemoryDispatchMarks_;
  std::vector<std::uint64_t> affectedMemoryServiceGroupMarks_;
  std::vector<std::uint64_t> affectedMemoryExposureMarks_;
  std::vector<std::uint64_t> affectedNetMarks_;
  std::vector<std::uint64_t> affectedBindingRelationMarks_;
  std::vector<PnrIndex> affectedComputes_;
  std::vector<PnrIndex> affectedMemories_;
  std::vector<PnrIndex> affectedPorts_;
  std::vector<PnrIndex> affectedBoundaries_;
  std::vector<PnrIndex> affectedMemoryPlans_;
  std::vector<PnrIndex> affectedLogicalMemories_;
  std::vector<PnrIndex> affectedMemoryDispatches_;
  std::vector<PnrIndex> affectedMemoryServiceGroups_;
  std::vector<PnrIndex> affectedMemoryExposures_;
  std::vector<PnrIndex> affectedNets_;
  std::vector<PnrIndex> affectedBindingRelations_;
  std::uint64_t affectedEpoch_ = 0;

  std::vector<PnrIndex> touchedRoutes_;
  std::vector<const RouteTreeState *> routeViews_;
  std::vector<llvm::ArrayRef<std::optional<llvm::APInt>>> tagValueViews_;
  std::vector<PnrIndex> physicalTimingChangedNets_;
  std::vector<std::uint64_t> physicalTimingOldWorstArrivals_;
  std::vector<std::uint64_t> physicalTimingOldNegativeSlacks_;
  std::vector<std::uint64_t> physicalTimingRouteNodeArrivals_;
  std::vector<std::pair<PnrIndex, std::uint64_t>>
      physicalTimingRouteNodeWorklist_;
  std::vector<PnrIndex> oldSwitchHandshakeFragments_;
  std::vector<PnrIndex> newSwitchHandshakeFragments_;
  std::vector<PnrIndex> removedSwitchHandshakeFragments_;
  std::vector<PnrIndex> addedSwitchHandshakeFragments_;
  bool switchHandshakeBaselineCaptured_ = false;

  std::vector<std::uint64_t> traversalDeltaMarks_;
  std::vector<PnrIndex> traversalRemoved_;
  std::vector<PnrIndex> traversalAdded_;
  std::vector<PnrIndex> touchedTraversals_;
  std::uint64_t traversalEpoch_ = 0;
  std::size_t resourceFullyAppliedRouteCount_ = 0;
  std::size_t resourcePartiallyAppliedDeltaCount_ = 0;

  std::vector<std::size_t> progressRecordedRouteDeltaCounts_;
  std::vector<std::uint64_t> progressRecordedRouteDeltaEpochs_;
  std::uint64_t progressRecordedRouteDeltaEpoch_ = 0;
  std::vector<std::uint8_t> progressTerminalActive_;
  std::vector<ProgressTraversalDelta> progressTraversalDeltas_;
  std::vector<std::uint8_t> progressDirtyNetMarks_;
  std::vector<PnrIndex> progressDirtyNets_;
  std::vector<std::uint64_t> progressDependencyJournalMarks_;
  std::vector<ProgressDependencyDelta> progressDependencyDeltas_;

  std::unique_ptr<detail::SpatialRouteConstraintScratch>
      routeConstraintScratch_;
  std::unique_ptr<detail::SpatialMemoryConstraintScratch>
      memoryConstraintScratch_;

  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
  SpatialMoveTransaction *activeTransaction_ = nullptr;

  friend class SpatialCandidateState;
  friend class SpatialMoveTransaction;
};

class SpatialCandidateState final
    : public std::enable_shared_from_this<SpatialCandidateState> {
public:
  static llvm::Expected<SpatialCandidateStateHandle>
  create(FrozenSpatialPnrProblemHandle problem,
         SpatialCandidateInitialization initialization);
  static llvm::Expected<SpatialCandidateStateHandle>
  create(const FrozenSpatialPnrProblem &,
         SpatialCandidateInitialization) = delete;

  /// Reconstructs an independent candidate from this candidate's selected
  /// decisions and complete RouteTrees. Rebuildable capacity, tag, handshake,
  /// timing, and objective state is derived again from its semantic owners.
  /// This is the snapshot boundary used for routed search incumbents.
  llvm::Expected<SpatialCandidateStateHandle> cloneFullyRouted() const;

  SpatialCandidateState(const SpatialCandidateState &) = delete;
  SpatialCandidateState(SpatialCandidateState &&) = delete;
  SpatialCandidateState &operator=(const SpatialCandidateState &) = delete;
  SpatialCandidateState &operator=(SpatialCandidateState &&) = delete;
  ~SpatialCandidateState() = default;

  const FrozenSpatialPnrProblem &problem() const { return *problem_; }
  const SpatialComputeBindingSelection &
  computeBinding(PnrIndex realization) const;
  const SpatialMemoryBindingSelection &
  memoryBinding(PnrIndex realization) const;
  PnrIndex portAttachment(PnrIndex demand) const;
  llvm::ArrayRef<PnrIndex> portAttachmentSelections() const {
    return portAttachments_;
  }
  PnrIndex graphBoundaryAttachment(PnrIndex boundary) const;
  llvm::ArrayRef<PnrIndex> graphBoundaryAttachmentSelections() const {
    return graphBoundaryAttachments_;
  }
  PnrIndex memoryOperationPlan(PnrIndex actor) const;
  const SpatialLogicalMemoryBindingSelection &
  logicalMemoryBinding(PnrIndex binding) const;
  PnrIndex memoryUseDispatch(PnrIndex use) const;
  PnrIndex memoryExposureSelection(PnrIndex exposure) const;
  PnrIndex registerFifoTransfer(PnrIndex logicalNet) const;
  bool usesRegisterFifo(PnrIndex logicalNet) const {
    return registerFifoTransfer(logicalNet) != getInvalidPnrIndex();
  }

  PnrIndex logicalNetSourceEndpoint(PnrIndex logicalNet) const;
  PnrIndex logicalNetSinkEndpoint(PnrIndex logicalNet,
                                  PnrIndex sinkObligation) const;
  std::uint32_t logicalNetPayloadWidth(PnrIndex logicalNet) const;
  const RouteTreeState &routeTree(PnrIndex logicalNet) const;
  const HandshakeCandidateState &handshake() const { return *handshake_; }
  std::uint64_t unroutedObligationCount() const {
    return unroutedObligationCount_;
  }
  std::uint64_t atomicCapacityOveruse() const { return atomicCapacityOveruse_; }
  std::uint64_t staticSchedulePressure() const {
    return staticSchedulePressure_;
  }
  std::uint64_t sharedOperandIngressPressure() const {
    return sharedOperandIngressPressure_;
  }
  std::uint64_t routeCapacityOveruse() const {
    return routeResources_.totalCapacityOveruseRaw();
  }
  std::uint64_t hardProgressViolation() const {
    return progressState_.hardProgressViolation();
  }
  bool hasTransportClosureViolation() const {
    return hardProgressViolation() != 0 || unroutedObligationCount() != 0 ||
           routeCapacityOveruse() != 0 || tagResidentCapacityOveruse() != 0 ||
           tagUnassignedCount() != 0 || tagConflictCount() != 0;
  }
  const SpatialProgressState &progress() const { return progressState_; }
  llvm::Expected<std::vector<SpatialFiniteBufferConflictWitness>>
  finiteBufferConflictWitnesses() const {
    return progressState_.finiteBufferConflictWitnesses(*this);
  }
  llvm::Error rebuildFiniteBufferConflictWitness(
      PnrIndex owner, SpatialFiniteBufferConflictWitness &witness) const {
    return progressState_.rebuildFiniteBufferConflictWitness(*this, owner,
                                                             witness);
  }
  /// Exact selected envelope cache. FrozenSpatialCapacityIndex remains the
  /// sole owner of envelope semantics; these dense views are rebuildable.
  PnrIndex resourceTimeEnvelopeRefcount(PnrIndex envelope) const;
  bool resourceTimeEnvelopeActive(PnrIndex envelope) const;
  PnrIndex activeResourceTimeEnvelopeCount() const {
    return activeResourceTimeEnvelopeCount_;
  }
  llvm::ArrayRef<std::uint64_t> activeResourceTimeEnvelopeBits() const {
    return activeResourceTimeEnvelopeBits_;
  }
  std::uint64_t resourceReleaseLatencyCycles() const {
    return resourceReleaseLatencyCycles_;
  }
  std::uint64_t resourceMinimumInitiationIntervalCycles() const {
    return resourceMinimumInitiationIntervalCycles_;
  }
  std::uint64_t routeReleaseLatencyCycles() const {
    return routeResources_.routeReleaseLatencyCycles();
  }
  std::uint64_t routeMinimumInitiationIntervalCycles() const {
    return routeResources_.routeMinimumInitiationIntervalCycles();
  }
  const SpatialRecurrenceTimingProjection &recurrenceTiming() const {
    return recurrenceTiming_;
  }
  std::uint64_t transportBitCycleDemand() const {
    return routeResources_.transportBitCycleDemand();
  }
  std::uint64_t worstRouteArrivalDelayQuanta() const {
    return worstRouteArrivalDelayQuanta_;
  }
  std::uint64_t totalRouteNegativeSlackQuanta() const {
    return totalRouteNegativeSlackQuanta_;
  }
  std::uint64_t totalSelectedTraversalClaim() const {
    return routeResources_.totalSelectedTraversalClaim();
  }
  PnrIndex routeClaimSelectionCount(PnrIndex claim) const {
    return routeResources_.routeClaimSelectionCount(claim);
  }
  PnrIndex logicalNetRouteClaimRefcount(PnrIndex logicalNet,
                                        PnrIndex claim) const {
    return routeResources_.logicalNetRouteClaimRefcount(logicalNet, claim);
  }
  llvm::ArrayRef<std::uint64_t>
  logicalNetRouteClaimBits(PnrIndex logicalNet) const {
    return routeResources_.logicalNetRouteClaimBits(logicalNet);
  }
  std::uint64_t routeCapacityUsageRaw(PnrIndex capacityDimension) const {
    return routeResources_.capacityUsageRaw(capacityDimension);
  }
  std::uint64_t routeCapacityOveruseRaw(PnrIndex capacityDimension) const {
    return routeResources_.capacityOveruseRaw(capacityDimension);
  }
  llvm::ArrayRef<SpatialTagContinuitySegment>
  tagSegments(PnrIndex logicalNet) const {
    return tagAssignments_.segments(logicalNet);
  }
  llvm::ArrayRef<std::optional<llvm::APInt>>
  tagValues(PnrIndex logicalNet) const {
    return tagAssignments_.values(logicalNet);
  }
  llvm::ArrayRef<PnrIndex> tagSegmentDomains(PnrIndex logicalNet,
                                             PnrIndex segment) const {
    return tagAssignments_.segmentDomains(logicalNet, segment);
  }
  std::uint64_t tagUnassignedCount() const {
    return tagAssignments_.unassignedCount();
  }
  std::uint64_t tagConflictCount() const {
    return tagAssignments_.conflictCount();
  }
  std::uint64_t tagResidentCapacityOveruse() const {
    return tagAssignments_.residentCapacityOveruse();
  }
  std::uint64_t tagDomainResidentCount(PnrIndex domain) const {
    return tagAssignments_.domainResidentCount(domain);
  }
  std::uint64_t tagDomainResidentCapacityOveruse(PnrIndex domain) const {
    return tagAssignments_.domainResidentCapacityOveruse(domain);
  }
  std::uint64_t tagDomainConflictCount(PnrIndex domain) const {
    return tagAssignments_.domainConflictCount(domain);
  }
  bool tagDomainValueConflicts(PnrIndex domain,
                               const llvm::APInt &value) const {
    return tagAssignments_.domainValueConflicts(domain, value);
  }
  llvm::Error verify() const;
  llvm::Expected<SpatialMoveTransaction>
  beginMove(SpatialCandidateScratch &scratch LLVM_LIFETIME_BOUND) &;
  llvm::Expected<SpatialMoveTransaction>
  beginMove(SpatialCandidateScratch &) && = delete;

private:
  SpatialCandidateState(
      FrozenSpatialPnrProblemHandle problem,
      std::vector<SpatialComputeBindingSelection> computeBindings,
      std::vector<SpatialMemoryBindingSelection> memoryBindings,
      std::vector<PnrIndex> bindingRelationChoices,
      std::vector<PnrIndex> portAttachments,
      std::vector<PnrIndex> graphBoundaryAttachments,
      std::vector<PnrIndex> memoryOperationPlans,
      std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryBindings,
      std::vector<PnrIndex> memoryUseDispatches,
      std::vector<PnrIndex> memoryExposureSelections,
      std::vector<PnrIndex> registerFifoTransfers,
      std::vector<RouteTreeStateHandle> routeTrees,
      HandshakeCandidateStateHandle handshake,
      SpatialRouteResourceState routeResources,
      SpatialTagAssignmentState tagAssignments,
      std::uint64_t unroutedObligationCount,
      std::uint64_t atomicCapacityOveruse, std::uint64_t staticSchedulePressure,
      std::uint64_t sharedOperandIngressPressure,
      std::vector<std::uint64_t> logicalNetWorstArrivalDelayQuanta,
      std::vector<std::uint64_t> logicalNetNegativeSlackQuanta,
      std::uint64_t worstRouteArrivalDelayQuanta,
      std::uint64_t totalRouteNegativeSlackQuanta)
      : problem_(std::move(problem)),
        computeBindings_(std::move(computeBindings)),
        memoryBindings_(std::move(memoryBindings)),
        bindingRelationChoices_(std::move(bindingRelationChoices)),
        portAttachments_(std::move(portAttachments)),
        graphBoundaryAttachments_(std::move(graphBoundaryAttachments)),
        memoryOperationPlans_(std::move(memoryOperationPlans)),
        logicalMemoryBindings_(std::move(logicalMemoryBindings)),
        memoryUseDispatches_(std::move(memoryUseDispatches)),
        memoryExposureSelections_(std::move(memoryExposureSelections)),
        registerFifoTransfers_(std::move(registerFifoTransfers)),
        routeTrees_(std::move(routeTrees)), handshake_(std::move(handshake)),
        routeResources_(std::move(routeResources)),
        tagAssignments_(std::move(tagAssignments)),
        unroutedObligationCount_(unroutedObligationCount),
        atomicCapacityOveruse_(atomicCapacityOveruse),
        staticSchedulePressure_(staticSchedulePressure),
        sharedOperandIngressPressure_(sharedOperandIngressPressure),
        logicalNetWorstArrivalDelayQuanta_(
            std::move(logicalNetWorstArrivalDelayQuanta)),
        logicalNetNegativeSlackQuanta_(
            std::move(logicalNetNegativeSlackQuanta)),
        worstRouteArrivalDelayQuanta_(worstRouteArrivalDelayQuanta),
        totalRouteNegativeSlackQuanta_(totalRouteNegativeSlackQuanta) {}

  llvm::Error validateComputeBinding(PnrIndex realization) const;
  llvm::Error validateMemoryBinding(PnrIndex realization) const;
  llvm::Error validatePortAttachment(PnrIndex demand) const;
  llvm::Error validateGraphBoundaryAttachment(PnrIndex boundary) const;
  llvm::Error validateMemoryOperationPlan(PnrIndex actor) const;
  llvm::Error validateLogicalMemoryBinding(PnrIndex binding) const;
  llvm::Error validateLogicalMemoryBindingOverlap(PnrIndex binding) const;
  llvm::Expected<bool>
  logicalMemoryBindingTargetSupported(PnrIndex binding, PnrIndex target) const;
  llvm::Expected<const FrozenSpatialMemoryDispatchDomain *>
  memoryDispatchDomain(PnrIndex use) const;
  llvm::Expected<bool>
  memoryUseDispatchSelectionSupported(PnrIndex use, PnrIndex selection) const;
  llvm::Error validateMemoryUseDispatch(PnrIndex use) const;
  llvm::Error validateMemoryExposureSelection(PnrIndex exposure) const;
  llvm::Error validateRegisterFifoTransfer(PnrIndex logicalNet) const;
  llvm::Error verifyRegisterFifoTransfers() const;
  llvm::Error verifyMemorySelections() const;
  llvm::Error rebuildMemoryServiceUsage();
  llvm::Error changeMemoryServiceUsage(PnrIndex use, PnrIndex oldOption,
                                       PnrIndex newOption);
  llvm::Error rebuildMemoryExposureUsage();
  void changeMemoryExposureUsage(PnrIndex exposure, PnrIndex oldOption,
                                 PnrIndex newOption);
  llvm::Error validateLogicalNet(PnrIndex logicalNet) const;
  llvm::Error verifyBindingRelations() const;
  llvm::Error verifyBindingRelation(PnrIndex relation) const;
  llvm::Error verifyHandshakeProjection() const;
  llvm::Expected<std::uint64_t> recomputeAtomicCapacityOveruse() const;
  llvm::Expected<std::vector<PnrIndex>>
  deriveResourceTimeEnvelopeRefcounts() const;
  llvm::Error rebuildResourceTimeEnvelopeSelections();
  llvm::Error verifyResourceTimeEnvelopeSelections() const;
  llvm::Error replaceResourceTimeEnvelopeSlice(PnrIndex oldOffset,
                                               PnrIndex oldCount,
                                               PnrIndex newOffset,
                                               PnrIndex newCount);
  llvm::Error replaceResourceTimeEnvelope(std::optional<PnrIndex> oldEnvelope,
                                          std::optional<PnrIndex> newEnvelope);
  llvm::Error applyResourceTimeEnvelopeDelta(PnrIndex envelope, bool add);
  llvm::Expected<PnrIndex>
  memoryServiceResourceTimeEnvelope(PnrIndex group, PnrIndex pattern) const;
  PnrIndex terminalEndpoint(FrozenSpatialTerminalBinding binding) const;
  std::uint32_t
  terminalPayloadWidth(FrozenSpatialTerminalBinding binding) const;
  llvm::Expected<SpatialCandidateRouteProjection> projectVerifiedRoutes(
      llvm::ArrayRef<const RouteTreeState *> routeTrees,
      SpatialTagAssignmentSummary *tagSummary,
      HandshakeProjectionScratch &handshakeProjectionScratch) const;
  llvm::Expected<SpatialTagAssignmentSummary>
  summarizeCurrentTagAssignments() const;
  llvm::Expected<SpatialTagAssignmentDelta> summarizeCurrentTagAssignmentDelta(
      llvm::ArrayRef<PnrIndex> logicalNets,
      llvm::ArrayRef<PnrIndex> changedDomains) const;

  FrozenSpatialPnrProblemHandle problem_;
  std::vector<SpatialComputeBindingSelection> computeBindings_;
  std::vector<SpatialMemoryBindingSelection> memoryBindings_;
  std::vector<PnrIndex> bindingRelationChoices_;
  std::vector<PnrIndex> portAttachments_;
  std::vector<PnrIndex> graphBoundaryAttachments_;
  std::vector<PnrIndex> memoryOperationPlans_;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryBindings_;
  std::vector<PnrIndex> memoryUseDispatches_;
  llvm::DenseMap<std::pair<PnrIndex, PnrIndex>, PnrIndex>
      memoryServicePatternRefcounts_;
  std::vector<PnrIndex> memoryServiceGroupActivePatternCounts_;
  std::vector<PnrIndex> resourceTimeEnvelopeRefcounts_;
  std::vector<std::uint64_t> activeResourceTimeEnvelopeBits_;
  PnrIndex activeResourceTimeEnvelopeCount_ = 0;
  std::uint64_t resourceReleaseLatencyCycles_ = 0;
  std::uint64_t resourceMinimumInitiationIntervalCycles_ = 1;
  std::vector<PnrIndex> memoryExposureSelections_;
  std::vector<PnrIndex> registerFifoTransfers_;
  llvm::DenseMap<std::pair<PnrIndex, PnrIndex>, PnrIndex>
      memoryExposureProviderRefcounts_;
  std::vector<PnrIndex> memoryExposureProviderBindingCounts_;
  std::vector<RouteTreeStateHandle> routeTrees_;
  HandshakeCandidateStateHandle handshake_;
  SpatialRouteResourceState routeResources_;
  SpatialProgressState progressState_;
  SpatialTagAssignmentState tagAssignments_;
  std::uint64_t unroutedObligationCount_ = 0;
  std::uint64_t atomicCapacityOveruse_ = 0;
  std::uint64_t staticSchedulePressure_ = 0;
  std::uint64_t sharedOperandIngressPressure_ = 0;
  std::vector<std::uint64_t> logicalNetWorstArrivalDelayQuanta_;
  std::vector<std::uint64_t> logicalNetNegativeSlackQuanta_;
  std::uint64_t worstRouteArrivalDelayQuanta_ = 0;
  std::uint64_t totalRouteNegativeSlackQuanta_ = 0;
  SpatialRecurrenceTimingProjection recurrenceTiming_;
  SpatialMoveTransaction *activeTransaction_ = nullptr;

  friend class SpatialActionDomainScratch;
  friend class SpatialActionExecutorScratch;
  friend class SpatialMoveTransaction;
};

class SpatialMoveTransaction final {
public:
  SpatialMoveTransaction(SpatialMoveTransaction &&other) noexcept;
  SpatialMoveTransaction(const SpatialMoveTransaction &) = delete;
  SpatialMoveTransaction &operator=(const SpatialMoveTransaction &) = delete;
  SpatialMoveTransaction &operator=(SpatialMoveTransaction &&) = delete;
  ~SpatialMoveTransaction();

  llvm::Error setComputeBinding(PnrIndex realization, PnrIndex placement,
                                PnrIndex instructionContext);
  llvm::Error setMemoryBinding(PnrIndex realization, PnrIndex placement);
  llvm::Error setPortAttachment(PnrIndex demand, PnrIndex attachmentOption);
  llvm::Error setGraphBoundaryAttachment(PnrIndex boundary,
                                         PnrIndex attachmentOption);
  llvm::Error setMemoryOperationPlan(PnrIndex actor, PnrIndex plan);
  llvm::Error setLogicalMemoryBinding(PnrIndex binding, PnrIndex target,
                                      std::uint64_t physicalOffsetBytes);
  llvm::Error setMemoryUseDispatch(PnrIndex use, PnrIndex dispatchOption);
  llvm::Error setMemoryExposureSelection(PnrIndex exposure,
                                         PnrIndex exposureOption);
  llvm::Error setRegisterFifoTransfer(PnrIndex logicalNet,
                                      std::optional<PnrIndex> option);

  llvm::Error bindRouteSource(PnrIndex logicalNet, PnrIndex endpoint);
  llvm::Error bindRouteSink(PnrIndex logicalNet, PnrIndex sinkObligation,
                            PnrIndex endpoint);
  llvm::Error attachRoutePath(PnrIndex logicalNet, PnrIndex attachmentEndpoint,
                              llvm::ArrayRef<PnrIndex> forwardArcs,
                              PnrIndex sinkObligation);
  llvm::Error ripUpRouteSink(PnrIndex logicalNet, PnrIndex sinkObligation);
  llvm::Error ripUpRouteSubtree(PnrIndex logicalNet,
                                PnrIndex subtreeRootEndpoint);
  llvm::Error ripUpWholeRoute(PnrIndex logicalNet);
  llvm::Expected<SpatialCandidateRouteProjection> projectCurrentRoutes();
  llvm::Expected<SpatialCandidateRouteProjection>
  projectCurrentRoutes(SpatialTagAssignmentSummary &tagSummary);

  llvm::Expected<bool> close();
  llvm::ArrayRef<PnrIndex> cycleWitness() const;
  llvm::ArrayRef<PnrIndex> touchedRouteTraversals() const;
  llvm::ArrayRef<PnrIndex> touchedRouteLogicalNets() const;
  llvm::Expected<SpatialTagAssignmentSummary>
  summarizeCurrentTagAssignments() const;
  llvm::Expected<SpatialTagAssignmentDelta>
  summarizeCurrentTagAssignmentDelta() const;
  bool hasRouteTreeChange() const;
  bool hasSemanticChange() const;
  llvm::Error commit();
  void rollback() noexcept;

private:
  SpatialMoveTransaction(SpatialCandidateStateHandle state,
                         SpatialCandidateScratch &scratch);

  llvm::Expected<SpatialCandidateRouteProjection>
  projectCurrentRoutesImpl(SpatialTagAssignmentSummary *tagSummary);
  llvm::Error ensureCollecting() const;
  llvm::Expected<RouteTreeTransaction *> routeTransaction(PnrIndex logicalNet);
  llvm::Error captureSwitchHandshakeBaseline();
  void rebuildRouteViews();
  void rebuildTagValueViews();
  void recordCompute(PnrIndex realization);
  void recordMemory(PnrIndex realization);
  void recordPort(PnrIndex demand);
  void recordBoundary(PnrIndex boundary);
  void recordMemoryPlan(PnrIndex actor);
  void recordLogicalMemory(PnrIndex binding);
  void recordMemoryDispatch(PnrIndex use);
  void recordMemoryExposure(PnrIndex exposure);
  void recordRegisterFifoTransfer(PnrIndex logicalNet);
  void markCompute(PnrIndex realization);
  void markMemory(PnrIndex realization);
  void markPort(PnrIndex demand);
  void markBoundary(PnrIndex boundary);
  void markMemoryPlan(PnrIndex actor);
  void markLogicalMemory(PnrIndex binding);
  void markMemoryDispatch(PnrIndex use);
  void markMemoryServiceGroup(PnrIndex group);
  void markMemoryExposure(PnrIndex exposure);
  void markNet(PnrIndex logicalNet);
  void markProgressNetDirty(PnrIndex logicalNet);
  void markBindingRelations(PnrIndex decision);
  llvm::Error changeFragments(llvm::ArrayRef<PnrIndex> oldFragments,
                              llvm::ArrayRef<PnrIndex> newFragments);
  llvm::Error changeTraversal(std::optional<PnrIndex> oldTraversal,
                              std::optional<PnrIndex> newTraversal);
  llvm::Error changeProgressTraversal(
      PnrIndex logicalNet, std::optional<PnrIndex> oldTraversal,
      std::optional<PnrIndex> newTraversal);
  llvm::Error changeProgressTerminalSelections(PnrIndex logicalNet,
                                               bool oldActive,
                                               bool newActive);
  llvm::Error
  changeRegisterFifoTransferResources(PnrIndex logicalNet,
                                      std::optional<PnrIndex> oldOption,
                                      std::optional<PnrIndex> newOption);
  llvm::Error recordTraversalSelectionDelta(PnrIndex traversal,
                                            PnrIndex removed, PnrIndex added);
  llvm::Error collectRouteTraversalDeltas();
  llvm::Error applyProgressTraversalDelta(PnrIndex logicalNet,
                                          PnrIndex traversal,
                                          PnrIndex removed, PnrIndex added);
  llvm::Error synchronizeProgressProjection();
  void rollbackProgressProjection() noexcept;
  void acceptProgressProjection() noexcept;
  void rollbackAppliedRouteResources() noexcept;
  void acceptAppliedRouteResources() noexcept;
  llvm::Error validateAffectedState() const;
  void finish();

  SpatialCandidateStateHandle state_;
  SpatialCandidateScratch *scratch_ = nullptr;
  bool closed_ = false;
  bool cycle_ = false;
  bool routeDeltasCollected_ = false;
  bool tagDeltasCollected_ = false;
  bool routeViolationApplied_ = false;
  std::uint64_t initialUnroutedObligationCount_ = 0;
  std::uint64_t initialAtomicCapacityOveruse_ = 0;
  std::uint64_t initialStaticSchedulePressure_ = 0;
  std::uint64_t initialWorstRouteArrivalDelayQuanta_ = 0;
  std::uint64_t initialTotalRouteNegativeSlackQuanta_ = 0;
  bool recurrenceTimingSelected_ = false;
  SpatialRecurrenceTimingProjection initialRecurrenceTiming_;
  friend class SpatialCandidateState;
  friend class SpatialCandidateScratch;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALCANDIDATESTATE_H
