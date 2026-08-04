#ifndef LOOM_PNR_SPATIALPATHFINDERROUTER_H
#define LOOM_PNR_SPATIALPATHFINDERROUTER_H

#include "PnR/SpatialNetRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace loom::pnr {

struct SpatialPathFinderRoutingLimits final {
  std::uint64_t endpointExpansionLimit = 0;
  std::uint64_t iterationLimit = 0;
};

struct SpatialPathFinderClosureResult final {
  std::uint64_t completedIterations = 0;
};

class SpatialPathFinderClosureFailure final
    : public llvm::ErrorInfo<SpatialPathFinderClosureFailure> {
public:
  enum class Kind {
    NonClosure,
    SelectedCombinationalHandshakeCycle,
  };

  static char ID;

  SpatialPathFinderClosureFailure(Kind kind, std::string message);

  Kind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
};

/// Worker-local deterministic PathFinder closure scratch. One invocation uses
/// one SpatialMoveTransaction for every routed net and iteration. A failed
/// invocation therefore rolls back the complete route overlay without a
/// second RouteTree snapshot. P/H and canonical net order are frozen within an
/// iteration; only a complete non-closed iteration advances P/H.
class SpatialPathFinderRouterScratch final {
public:
  SpatialPathFinderRouterScratch() = default;
  SpatialPathFinderRouterScratch(const SpatialPathFinderRouterScratch &) =
      delete;
  SpatialPathFinderRouterScratch &
  operator=(const SpatialPathFinderRouterScratch &) = delete;
  SpatialPathFinderRouterScratch(SpatialPathFinderRouterScratch &&) = delete;
  SpatialPathFinderRouterScratch &
  operator=(SpatialPathFinderRouterScratch &&) = delete;
  ~SpatialPathFinderRouterScratch() = default;

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);

  llvm::Expected<SpatialPathFinderClosureResult>
  routeToClosure(SpatialCandidateState &candidate,
                 SpatialCandidateScratch &candidateScratch,
                 SpatialRouteCostState &costs,
                 SpatialPathFinderRoutingLimits limits,
                 llvm::ArrayRef<RouteCost> evaluationPriorities);

  /// Applies global negotiation inside an already active Mapping move. The
  /// caller owns close, objective evaluation, and commit or rollback.
  llvm::Expected<SpatialPathFinderClosureResult> routeToClosureInMove(
      SpatialMoveTransaction &move, SpatialCandidateState &candidate,
      SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
      llvm::ArrayRef<RouteCost> evaluationPriorities);

  llvm::Expected<RouteCost>
  routeWholeNetInMove(SpatialMoveTransaction &move,
                      const SpatialCandidateState &candidate,
                      SpatialRouteCostState &costs, PnrIndex logicalNet,
                      std::uint64_t endpointExpansionLimit);

  llvm::Error beginConstraintSweep(llvm::ArrayRef<PnrIndex> logicalNets);
  llvm::Error finishConstraintNet(PnrIndex logicalNet);

  std::size_t retainedStorageBytes() const;

private:
  struct NetOrderEntry final {
    std::uint8_t routeStateRank = 0;
    RouteCost conflictPressure = 0;
    RouteCost evaluationPriority = 0;
    PnrIndex logicalNet = 0;
  };

  struct NetProjection final {
    std::uint8_t routeStateRank = 0;
    RouteCost conflictPressure = 0;
  };

  llvm::Expected<NetProjection>
  projectLogicalNet(const SpatialCandidateState &candidate,
                    const SpatialRouteCostState &costs, PnrIndex logicalNet);
  llvm::Error
  buildCanonicalNetOrder(const SpatialCandidateState &candidate,
                         const SpatialRouteCostState &costs,
                         llvm::ArrayRef<RouteCost> evaluationPriorities);
  void beginProjection();

  SpatialNetRouterScratch netRouter_;
  std::vector<NetOrderEntry> netOrder_;
  std::vector<std::uint64_t> activeClaimBits_;
  std::vector<std::uint64_t> claimEpochs_;
  std::vector<std::uint64_t> capacityEpochs_;
  std::vector<RouteCost> capacityNetQCosts_;
  std::vector<PnrIndex> touchedCapacities_;
  std::vector<PnrIndex> constraintSweepNets_;
  std::uint64_t projectionEpoch_ = 0;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPATHFINDERROUTER_H
