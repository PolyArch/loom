#ifndef LOOM_PNR_SPATIALCANDIDATESTATE_H
#define LOOM_PNR_SPATIALCANDIDATESTATE_H

#include "PnR/HandshakeCandidateState.h"
#include "PnR/RouteTreeState.h"
#include "PnR/SpatialPnrProblem.h"
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
};

class SpatialCandidateState;
class SpatialCandidateScratch;
class SpatialActionDomainScratch;
class SpatialActionExecutorScratch;
class SpatialMoveTransaction;

using SpatialCandidateStateHandle = std::shared_ptr<SpatialCandidateState>;

class SpatialCandidateScratch final {
public:
  SpatialCandidateScratch() = default;
  SpatialCandidateScratch(const SpatialCandidateScratch &) = delete;
  SpatialCandidateScratch &operator=(const SpatialCandidateScratch &) = delete;
  SpatialCandidateScratch(SpatialCandidateScratch &&) = delete;
  SpatialCandidateScratch &operator=(SpatialCandidateScratch &&) = delete;
  ~SpatialCandidateScratch();

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);
  std::size_t retainedStorageBytes() const;

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
  };

  struct DecisionDelta final {
    DecisionKind kind = DecisionKind::ComputeBinding;
    PnrIndex index = 0;
    PnrIndex oldValue0 = 0;
    PnrIndex oldValue1 = 0;
    PnrIndex oldValue2 = 0;
    std::uint64_t oldWideValue = 0;
  };

  void beginTransaction();
  void resetTransaction();

  std::vector<std::unique_ptr<RouteTreeTransactionScratch>> routeScratch_;
  std::vector<std::optional<RouteTreeTransaction>> routeTransactions_;
  SpatialTagAssignmentScratch tagScratch_;
  HandshakeCandidateScratch handshakeScratch_;
  std::optional<HandshakeCandidateTransaction> handshakeTransaction_;

  std::vector<std::uint64_t> computeJournalMarks_;
  std::vector<std::uint64_t> memoryJournalMarks_;
  std::vector<std::uint64_t> portJournalMarks_;
  std::vector<std::uint64_t> boundaryJournalMarks_;
  std::vector<std::uint64_t> memoryPlanJournalMarks_;
  std::vector<std::uint64_t> logicalMemoryJournalMarks_;
  std::vector<std::uint64_t> memoryDispatchJournalMarks_;
  std::vector<std::uint64_t> memoryExposureJournalMarks_;
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

  std::vector<std::uint64_t> traversalDeltaMarks_;
  std::vector<PnrIndex> traversalRemoved_;
  std::vector<PnrIndex> traversalAdded_;
  std::vector<PnrIndex> touchedTraversals_;
  std::uint64_t traversalEpoch_ = 0;
  std::size_t resourceFullyAppliedRouteCount_ = 0;
  std::size_t resourcePartiallyAppliedDeltaCount_ = 0;

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
  PnrIndex graphBoundaryAttachment(PnrIndex boundary) const;
  PnrIndex memoryOperationPlan(PnrIndex actor) const;
  const SpatialLogicalMemoryBindingSelection &
  logicalMemoryBinding(PnrIndex binding) const;
  PnrIndex memoryUseDispatch(PnrIndex use) const;
  PnrIndex memoryExposureSelection(PnrIndex exposure) const;

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
  std::uint64_t routeCapacityOveruse() const {
    return routeResources_.totalCapacityOveruseRaw();
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
      std::vector<RouteTreeStateHandle> routeTrees,
      HandshakeCandidateStateHandle handshake,
      SpatialRouteResourceState routeResources,
      SpatialTagAssignmentState tagAssignments,
      std::uint64_t unroutedObligationCount,
      std::uint64_t atomicCapacityOveruse)
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
        routeTrees_(std::move(routeTrees)), handshake_(std::move(handshake)),
        routeResources_(std::move(routeResources)),
        tagAssignments_(std::move(tagAssignments)),
        unroutedObligationCount_(unroutedObligationCount),
        atomicCapacityOveruse_(atomicCapacityOveruse) {}

  llvm::Error validateComputeBinding(PnrIndex realization) const;
  llvm::Error validateMemoryBinding(PnrIndex realization) const;
  llvm::Error validatePortAttachment(PnrIndex demand) const;
  llvm::Error validateGraphBoundaryAttachment(PnrIndex boundary) const;
  llvm::Error validateMemoryOperationPlan(PnrIndex actor) const;
  llvm::Error validateLogicalMemoryBinding(PnrIndex binding) const;
  llvm::Error validateLogicalMemoryBindingOverlap(PnrIndex binding) const;
  llvm::Expected<const FrozenSpatialMemoryDispatchDomain *>
  memoryDispatchDomain(PnrIndex use) const;
  llvm::Error validateMemoryUseDispatch(PnrIndex use) const;
  llvm::Error validateMemoryExposureSelection(PnrIndex exposure) const;
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
  void applyResourceTimeEnvelopeDelta(PnrIndex envelope, bool add) noexcept;
  llvm::Expected<PnrIndex>
  memoryServiceResourceTimeEnvelope(PnrIndex group, PnrIndex pattern) const;
  PnrIndex terminalEndpoint(FrozenSpatialTerminalBinding binding) const;
  std::uint32_t
  terminalPayloadWidth(FrozenSpatialTerminalBinding binding) const;

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
  std::vector<PnrIndex> memoryExposureSelections_;
  llvm::DenseMap<std::pair<PnrIndex, PnrIndex>, PnrIndex>
      memoryExposureProviderRefcounts_;
  std::vector<PnrIndex> memoryExposureProviderBindingCounts_;
  std::vector<RouteTreeStateHandle> routeTrees_;
  HandshakeCandidateStateHandle handshake_;
  SpatialRouteResourceState routeResources_;
  SpatialTagAssignmentState tagAssignments_;
  std::uint64_t unroutedObligationCount_ = 0;
  std::uint64_t atomicCapacityOveruse_ = 0;
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

  llvm::Expected<bool> close();
  llvm::ArrayRef<PnrIndex> cycleWitness() const;
  llvm::ArrayRef<PnrIndex> touchedRouteTraversals() const;
  llvm::Error commit();
  void rollback() noexcept;

private:
  SpatialMoveTransaction(SpatialCandidateStateHandle state,
                         SpatialCandidateScratch &scratch);

  llvm::Error ensureCollecting() const;
  llvm::Expected<RouteTreeTransaction *> routeTransaction(PnrIndex logicalNet);
  void recordCompute(PnrIndex realization);
  void recordMemory(PnrIndex realization);
  void recordPort(PnrIndex demand);
  void recordBoundary(PnrIndex boundary);
  void recordMemoryPlan(PnrIndex actor);
  void recordLogicalMemory(PnrIndex binding);
  void recordMemoryDispatch(PnrIndex use);
  void recordMemoryExposure(PnrIndex exposure);
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
  void markBindingRelations(PnrIndex decision);
  llvm::Error changeFragments(llvm::ArrayRef<PnrIndex> oldFragments,
                              llvm::ArrayRef<PnrIndex> newFragments);
  llvm::Error changeTraversal(std::optional<PnrIndex> oldTraversal,
                              std::optional<PnrIndex> newTraversal);
  llvm::Error collectRouteTraversalDeltas();
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

  friend class SpatialCandidateState;
  friend class SpatialCandidateScratch;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALCANDIDATESTATE_H
