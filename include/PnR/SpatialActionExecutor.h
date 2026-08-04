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
}

class SpatialActionExecutorScratch;

enum class SpatialActionTransitionFailureKind : std::uint8_t {
  IntrinsicInvalid,
  WorkLimit,
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
                     bool globalRouting);

  SpatialActionExecutorScratch *owner_ = nullptr;
  SpatialMoveTransaction move_;
  dse::ObjectiveVector objective_;
  dse::ObjectiveSignedDifference energyDifference_;
  bool globalRouting_ = false;

  friend class SpatialActionExecutorScratch;
};

/// Worker-local executor for the closed Spatial Action algebra. It owns one
/// candidate transaction scratch, one shared local/global router scratch, and
/// one removable route-cost projection for the exact candidate.
class SpatialActionExecutorScratch final {
public:
  SpatialActionExecutorScratch();
  ~SpatialActionExecutorScratch();

  llvm::Error prepare(SpatialCandidateState &candidate);
  llvm::Expected<SpatialActionProbe> probe(SpatialCandidateState &candidate,
                                           const SpatialMappingAction &action);
  llvm::Expected<SpatialActionProbe>
  probeBatch(SpatialCandidateState &candidate,
             llvm::ArrayRef<SpatialMappingAction> actions);

  const dse::ObjectiveVector &currentObjective() const;
  std::size_t retainedStorageBytes() const;

private:
  llvm::Error apply(SpatialMoveTransaction &move,
                    SpatialCandidateState &candidate,
                    const SpatialMappingAction &action);
  llvm::Error applyComputeBinding(SpatialMoveTransaction &move,
                                  SpatialCandidateState &candidate,
                                  SpatialComputeBindingAction action);
  llvm::Error applyMemoryBinding(SpatialMoveTransaction &move,
                                 SpatialCandidateState &candidate,
                                 SpatialMemoryBindingAction action);
  llvm::Error routeAffectedNets(SpatialMoveTransaction &move,
                                SpatialCandidateState &candidate);
  llvm::Error reconcileBindingRelations(SpatialMoveTransaction &move,
                                        SpatialCandidateState &candidate);
  void markChangedBindingRoot(PnrIndex decision);
  void markExplicitAttachment(PnrIndex decision);
  llvm::Error markNet(PnrIndex logicalNet);
  void beginDependencyClosure();
  llvm::Error restoreAfterFailure(SpatialMoveTransaction &move,
                                  llvm::Error failure);

  SpatialCandidateScratch candidateScratch_;
  SpatialPathFinderRouterScratch router_;
  std::optional<SpatialRouteCostState> routeCosts_;
  std::optional<dse::ObjectiveVector> currentObjective_;
  std::vector<std::uint64_t> netMarks_;
  std::vector<PnrIndex> affectedNets_;
  std::vector<PnrIndex> routeCostTraversals_;
  std::unique_ptr<detail::InitializerRelationSolver> relationSolver_;
  std::vector<PnrIndex> fixedRelationChoices_;
  std::vector<std::uint8_t> relationDecisionMarks_;
  std::vector<std::uint8_t> explicitAttachmentMarks_;
  std::vector<PnrIndex> relationDecisionQueue_;
  std::vector<PnrIndex> changedBindingRoots_;
  std::uint64_t netEpoch_ = 0;
  SpatialCandidateState *candidate_ = nullptr;
  bool activeProbe_ = false;
  bool globalRouting_ = false;

  friend class SpatialActionProbe;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALACTIONEXECUTOR_H
