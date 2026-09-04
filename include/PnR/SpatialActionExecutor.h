#ifndef LOOM_PNR_SPATIALACTIONEXECUTOR_H
#define LOOM_PNR_SPATIALACTIONEXECUTOR_H

#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/SpatialAction.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPathFinderRouter.h"
#include "PnR/SpatialRouteCostState.h"

#include "DSE/Objective.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::pnr {

namespace detail {
class InitializerRelationSolver;
class SpatialMemoryConstraintScratch;
} // namespace detail

class SpatialActionExecutorScratch;

enum class SpatialActionTransitionFailureKind : std::uint8_t {
  IntrinsicInvalid,
  WorkLimit,
  Interrupted,
};

enum class SpatialActionExecutionContext : std::uint8_t {
  Search,
  ExactRepair,
  FinalClosure,
};

/// Exact-repair selection of one literal from the frozen canonical no-good.
/// The pair is a lookup key only: it never duplicates the literal's Mapping
/// or Fabric identity and is invalid outside an ExactRepair probe.
struct SpatialRuntimeLiteralBreaker final {
  PnrIndex clauseOrdinal = 0;
  PnrIndex clauseLocalLiteralOrdinal = 0;
};

/// A well-formed Action that cannot produce a candidate transition. Search
/// consumes the proposal slot and continues; malformed Actions and owner
/// invariant failures use their original errors and terminate the invocation.
class SpatialActionTransitionFailure final
    : public llvm::ErrorInfo<SpatialActionTransitionFailure> {
public:
  static char ID;

  SpatialActionTransitionFailure(SpatialActionTransitionFailureKind kind,
                                 std::string message)
      : kind_(kind), message_(std::move(message)) {}

  SpatialActionTransitionFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SpatialActionTransitionFailureKind kind_;
  std::string message_;
};

struct SpatialActionResolution final {
  bool accepted = false;
  dse::ObjectiveVector objective;
};

/// One closed Mapping shadow transition. The candidate and route-cost overlay
/// remain provisional until commit or discard resolves this object.
class SpatialActionProbe final {
public:
  SpatialActionProbe(SpatialActionProbe &&other) noexcept;
  SpatialActionProbe(const SpatialActionProbe &) = delete;
  SpatialActionProbe &operator=(const SpatialActionProbe &) = delete;
  SpatialActionProbe &operator=(SpatialActionProbe &&) = delete;
  ~SpatialActionProbe();

  const dse::ObjectiveVector &objective() const { return objective_; }
  dse::ObjectiveSignedDifference energyDifference() const {
    return energyDifference_;
  }
  bool isSemanticNoop() const { return !semanticChange_; }

  llvm::Error commit();
  llvm::Error discard();
  llvm::Expected<SpatialActionResolution>
  resolve(std::uint64_t temperature,
          DeterministicPnrRandomStream &acceptanceStream);

private:
  SpatialActionProbe(SpatialActionExecutorScratch &owner,
                     SpatialMoveTransaction move,
                     dse::ObjectiveVector objective,
                     dse::ObjectiveSignedDifference energyDifference,
                     bool negotiatedRouting, bool routeTagsSynchronized,
                     bool semanticChange);

  SpatialActionExecutorScratch *owner_ = nullptr;
  SpatialMoveTransaction move_;
  dse::ObjectiveVector objective_;
  dse::ObjectiveSignedDifference energyDifference_;
  bool negotiatedRouting_ = false;
  bool routeTagsSynchronized_ = false;
  bool semanticChange_ = true;

  friend class SpatialActionExecutorScratch;
};

/// Worker-local executor for the closed Spatial Action algebra. It owns one
/// candidate transaction scratch, one shared local/global router scratch, and
/// one removable route-cost projection for the exact candidate.
class SpatialActionExecutorScratch final {
public:
  SpatialActionExecutorScratch();
  ~SpatialActionExecutorScratch();

  llvm::Error prepare(SpatialCandidateState &candidate,
                      SpatialPnrWorkLedgerView workLedger = {},
                      ExecutionControlView executionControl = {});
  llvm::Expected<SpatialActionProbe>
  probe(SpatialCandidateState &candidate, const SpatialMappingAction &action,
        SpatialActionExecutionContext context =
            SpatialActionExecutionContext::Search);
  llvm::Expected<SpatialActionProbe>
  probeBatch(SpatialCandidateState &candidate,
             llvm::ArrayRef<SpatialMappingAction> actions,
             SpatialActionExecutionContext context =
                 SpatialActionExecutionContext::Search,
             std::uint64_t exactRegionalLogicalNetLimit = 0,
             std::optional<SpatialRuntimeLiteralBreaker> runtimeBreaker =
                 std::nullopt);

  const dse::ObjectiveVector &currentObjective() const;
  /// Coldly reconstructs the bound candidate's route-cost projection.
  llvm::Error verifyCandidateProjection() const;
  std::uint64_t endpointExpansionCount() const {
    return router_.endpointExpansionCount();
  }
  std::uint64_t heuristicCacheHitCount() const {
    return router_.heuristicCacheHitCount();
  }
  std::uint64_t heuristicBuildCount() const {
    return router_.heuristicBuildCount();
  }
  std::uint64_t forwardHeuristicQueryCount() const {
    return router_.forwardHeuristicQueryCount();
  }
  std::uint64_t forwardHeuristicUnreachableCount() const {
    return router_.forwardHeuristicUnreachableCount();
  }
  std::uint64_t heuristicCacheEvictionCount() const {
    return router_.heuristicCacheEvictionCount();
  }
  std::uint64_t arcCostValidationScanCount() const {
    return router_.arcCostValidationScanCount();
  }
  std::uint64_t physicalTimingValidationScanCount() const {
    return router_.physicalTimingValidationScanCount();
  }
  std::size_t heuristicCacheEntryCount() const {
    return router_.heuristicCacheEntryCount();
  }
  std::size_t heuristicCacheRetainedBytes() const {
    return router_.heuristicCacheRetainedBytes();
  }
  std::uint64_t negotiationIterationCount() const {
    return router_.negotiationIterationCount();
  }
  HandshakeProjectionStatistics handshakeProjectionStatistics() const {
    return candidateScratch_.handshakeProjectionStatistics();
  }
  std::uint64_t regionalLogicalNetCount() const {
    return router_.regionalLogicalNetCount();
  }
  llvm::ArrayRef<PnrIndex> regionalLogicalNets() const {
    return router_.regionalLogicalNets();
  }
  /// The decision and touched-net change sets of the most recently committed
  /// probe, for incremental Action-domain maintenance.
  bool hasCommittedChanges() const { return committedChangesValid_; }
  llvm::ArrayRef<std::pair<SpatialCandidateScratch::DecisionKind, PnrIndex>>
  committedDecisionChanges() const {
    return committedDecisionChanges_;
  }
  llvm::ArrayRef<PnrIndex> committedLogicalNetChanges() const {
    return committedLogicalNetChanges_;
  }
  std::size_t retainedStorageBytes() const;

private:
  enum class PendingRouteKind : std::uint8_t {
    WholeNet,
    SingleSink,
    RootedSubtree,
  };

  llvm::Error apply(SpatialMoveTransaction &move,
                    SpatialCandidateState &candidate,
                    const SpatialMappingAction &action);
  llvm::Error applyComputeBinding(SpatialMoveTransaction &move,
                                  SpatialCandidateState &candidate,
                                  SpatialComputeBindingAction action);
  llvm::Error applyMemoryBinding(SpatialMoveTransaction &move,
                                 SpatialCandidateState &candidate,
                                 SpatialMemoryBindingAction action);
  llvm::Error reconcileLogicalMemoryBinding(SpatialMoveTransaction &move,
                                            SpatialCandidateState &candidate,
                                            PnrIndex binding);
  void markChangedLogicalMemoryBinding(PnrIndex binding);
  llvm::Error
  recordExplicitLogicalMemoryBinding(const SpatialCandidateState &candidate,
                                     SpatialLogicalMemoryBindingAction action);
  llvm::Expected<bool>
  explicitLogicalMemoryTargetSupported(const SpatialCandidateState &candidate,
                                       PnrIndex binding, PnrIndex target) const;
  llvm::Error
  reconcileExplicitLogicalMemoryBindings(SpatialMoveTransaction &move,
                                         SpatialCandidateState &candidate);
  llvm::Error
  recordExplicitMemoryDispatch(const SpatialCandidateState &candidate,
                               PnrIndex use, PnrIndex option);
  llvm::Error
  reconcileExplicitMemoryDispatches(SpatialMoveTransaction &move,
                                    SpatialCandidateState &candidate);
  llvm::Error
  recordExplicitMemoryExposure(const SpatialCandidateState &candidate,
                               PnrIndex exposure, PnrIndex option);
  llvm::Error
  reconcileExplicitMemoryExposures(SpatialMoveTransaction &move,
                                   SpatialCandidateState &candidate);
  llvm::Error routeAffectedNets(SpatialMoveTransaction &move,
                                SpatialCandidateState &candidate);
  llvm::Error realizeExplicitLocalDispositions(
      SpatialMoveTransaction &move, SpatialCandidateState &candidate);
  llvm::Error reconcileBindingRelations(SpatialMoveTransaction &move,
                                        SpatialCandidateState &candidate);
  void markChangedBindingRoot(PnrIndex decision);
  void markExplicitAttachment(PnrIndex decision);
  llvm::Error markNet(PnrIndex logicalNet);
  llvm::Error markWholeNet(SpatialWholeNetRoutingAction action);
  llvm::Error markLocalNet(PnrIndex logicalNet, PendingRouteKind kind,
                           PnrIndex localAnchor);
  llvm::Error markWitnessRegion(SpatialMoveTransaction &move,
                                SpatialWitnessRegionRoutingAction action);
  llvm::Error selectExternalAttachments(SpatialMoveTransaction &move,
                                        SpatialCandidateState &candidate,
                                        PnrIndex logicalNet);
  void beginDependencyClosure();
  llvm::Error restoreAfterFailure(SpatialMoveTransaction &move,
                                  llvm::Error failure,
                                  bool resetNegotiationState);
  llvm::Error
  synchronizeCandidateTags(llvm::ArrayRef<PnrIndex> changedLogicalNets);
  llvm::Error restoreCandidateTagDelta();

  SpatialCandidateScratch candidateScratch_;
  SpatialPathFinderRouterScratch router_;
  std::optional<SpatialRouteCostState> routeCosts_;
  std::optional<dse::ObjectiveVector> currentObjective_;
  std::vector<std::uint64_t> netMarks_;
  std::vector<PendingRouteKind> pendingRouteKinds_;
  std::vector<PnrIndex> pendingRouteAnchors_;
  std::vector<std::uint8_t> explicitNetDispositionMarks_;
  std::vector<SpatialWholeNetDispositionKind> explicitNetDispositions_;
  std::vector<PnrIndex> explicitRegisterFifoTransfers_;
  std::vector<PnrIndex> affectedNets_;
  SpatialFiniteBufferConflictWitness capacityProofDebtWitness_;
  std::vector<PnrIndex> routeCostTraversals_;
  std::vector<PnrIndex> routeCostLogicalNets_;
  std::vector<std::pair<SpatialCandidateScratch::DecisionKind, PnrIndex>>
      committedDecisionChanges_;
  std::vector<PnrIndex> committedLogicalNetChanges_;
  bool committedChangesValid_ = false;
  std::vector<PnrIndex> routeTagLogicalNets_;
  std::vector<PnrIndex> routeTagDomains_;
  std::vector<std::uint64_t> localTransferClaimBits_;
  std::vector<PnrIndex> localTransferClaimWords_;
  std::unique_ptr<detail::InitializerRelationSolver> relationSolver_;
  std::unique_ptr<detail::SpatialMemoryConstraintScratch>
      memoryConstraintScratch_;
  std::vector<PnrIndex> fixedRelationChoices_;
  std::vector<std::uint8_t> relationDecisionMarks_;
  std::vector<std::uint8_t> explicitAttachmentMarks_;
  std::vector<PnrIndex> relationDecisionQueue_;
  std::vector<PnrIndex> releasedRelationDecisions_;
  std::vector<PnrIndex> changedBindingRoots_;
  std::vector<SpatialLogicalMemoryBindingSelection>
      explicitLogicalMemorySelections_;
  std::vector<std::uint8_t> explicitLogicalMemoryMarks_;
  std::vector<PnrIndex> explicitLogicalMemoryBindings_;
  std::vector<SpatialLogicalMemoryBindingSelection>
      explicitLogicalMemoryChoices_;
  std::vector<std::uint8_t> changedLogicalMemoryMarks_;
  std::vector<PnrIndex> changedLogicalMemoryBindings_;
  std::vector<PnrIndex> explicitMemoryDispatchPatterns_;
  std::vector<std::uint8_t> explicitMemoryDispatchGroupMarks_;
  std::vector<PnrIndex> explicitMemoryDispatchGroups_;
  std::vector<PnrIndex> explicitMemoryDispatchSelections_;
  std::vector<std::uint8_t> explicitMemoryDispatchUseMarks_;
  std::vector<PnrIndex> explicitMemoryExposureSelections_;
  std::vector<std::uint8_t> explicitMemoryExposureMarks_;
  std::uint64_t netEpoch_ = 0;
  std::uint8_t dependencyEpoch_ = 0;
  SpatialCandidateState *candidate_ = nullptr;
  bool activeProbe_ = false;
  bool globalRouting_ = false;

  friend class SpatialActionProbe;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALACTIONEXECUTOR_H
