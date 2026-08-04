#include "PnR/SpatialPathFinderRouter.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace loom::pnr;

char SpatialPathFinderClosureFailure::ID;

SpatialPathFinderClosureFailure::SpatialPathFinderClosureFailure(
    Kind kind, std::string message)
    : kind_(kind), message_(std::move(message)) {}

void SpatialPathFinderClosureFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SpatialPathFinderClosureFailure::convertToErrorCode() const {
  switch (kind_) {
  case Kind::NonClosure:
    return std::make_error_code(std::errc::resource_unavailable_try_again);
  case Kind::SelectedCombinationalHandshakeCycle:
    return std::make_error_code(std::errc::state_not_recoverable);
  }
  llvm_unreachable("invalid Spatial PathFinder closure failure kind");
}

namespace {

llvm::Error pathFinderError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial PathFinder route: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

llvm::Error rollbackIteration(SpatialMoveTransaction &move,
                              SpatialRouteCostState &costs,
                              llvm::Error failure) {
  move.rollback();
  if (llvm::Error reset = costs.resetFromCandidate())
    return llvm::joinErrors(std::move(failure), std::move(reset));
  return failure;
}

} // namespace

llvm::Error SpatialPathFinderRouterScratch::prepare(
    const FrozenSpatialPnrProblem &problem) {
  if (llvm::Error error = netRouter_.prepare(problem))
    return error;
  const std::size_t logicalNetCount = problem.transfers().logicalNets().size();
  const std::size_t routeClaimCount = problem.routing().routeClaims().size();
  const std::size_t capacityCount =
      problem.resources().capacityDimensions().size();

  netOrder_.clear();
  netOrder_.reserve(logicalNetCount);
  activeClaimBits_.assign((routeClaimCount + 63) / 64, 0);
  claimEpochs_.assign(routeClaimCount, 0);
  capacityEpochs_.assign(capacityCount, 0);
  capacityNetQCosts_.assign(capacityCount, 0);
  touchedCapacities_.clear();
  touchedCapacities_.reserve(capacityCount);
  constraintSweepNets_.clear();
  constraintSweepNets_.reserve(logicalNetCount);
  projectionEpoch_ = 0;
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

void SpatialPathFinderRouterScratch::beginProjection() {
  ++projectionEpoch_;
  if (projectionEpoch_ == 0) {
    std::fill(claimEpochs_.begin(), claimEpochs_.end(), 0);
    std::fill(capacityEpochs_.begin(), capacityEpochs_.end(), 0);
    projectionEpoch_ = 1;
  }
  std::fill(activeClaimBits_.begin(), activeClaimBits_.end(), 0);
  touchedCapacities_.clear();
}

llvm::Expected<SpatialPathFinderRouterScratch::NetProjection>
SpatialPathFinderRouterScratch::projectLogicalNet(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    PnrIndex logicalNet) {
  if (logicalNet >= candidate.problem().transfers().logicalNets().size())
    return pathFinderError("logical net is out of range");
  beginProjection();

  const RouteTreeState &tree = candidate.routeTree(logicalNet);
  if (tree.isUnrouted())
    return NetProjection{0, 0};

  const FrozenSpatialRoutingGraph &routing = candidate.problem().routing();
  for (const RouteTreeNode &node : tree.nodeStorage()) {
    if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
      continue;
    if (node.parentArc >= routing.routingArcs().size())
      return pathFinderError("RouteTree parent arc is out of range");
    const PnrIndex traversal = routing.routingArcs()[node.parentArc].traversal;
    if (traversal >= routing.traversals().size())
      return pathFinderError("RouteTree traversal is out of range");
    const FrozenSpatialTraversal &record = routing.traversals()[traversal];
    for (PnrIndex claim : routing.traversalClaimKeys().slice(
             record.routeClaimOffset, record.routeClaimCount)) {
      if (claim >= routing.routeClaims().size())
        return pathFinderError("RouteTree claim is out of range");
      if (claimEpochs_[claim] == projectionEpoch_)
        continue;
      claimEpochs_[claim] = projectionEpoch_;
      activeClaimBits_[claim / 64] |= std::uint64_t{1} << (claim % 64);

      const FrozenSpatialRouteClaim &claimRecord = routing.routeClaims()[claim];
      const PnrIndex capacity = claimRecord.capacityDimension;
      if (capacity >= capacityNetQCosts_.size())
        return pathFinderError("RouteTree claim capacity is out of range");
      if (capacityEpochs_[capacity] != projectionEpoch_) {
        capacityEpochs_[capacity] = projectionEpoch_;
        capacityNetQCosts_[capacity] = 0;
        touchedCapacities_.push_back(capacity);
      }
      auto qCost =
          accumulateRouteCost(capacityNetQCosts_[capacity], claimRecord.qCost);
      if (!qCost)
        return qCost.takeError();
      capacityNetQCosts_[capacity] = *qCost;
    }
  }

  RouteCost conflictPressure = 0;
  bool contributesToViolation = false;
  for (PnrIndex capacity : touchedCapacities_) {
    const RouteCost overuse = costs.capacityOveruseCost(capacity);
    contributesToViolation |= overuse != 0;
    auto term = scaledRouteProduct(capacityNetQCosts_[capacity], overuse);
    if (!term)
      return term.takeError();
    auto next = accumulateRouteCost(conflictPressure, *term);
    if (!next)
      return next.takeError();
    conflictPressure = *next;
  }
  return NetProjection{
      static_cast<std::uint8_t>(contributesToViolation ? 1 : 2),
      conflictPressure};
}

llvm::Error SpatialPathFinderRouterScratch::buildCanonicalNetOrder(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  const std::size_t logicalNetCount =
      candidate.problem().transfers().logicalNets().size();
  if (!evaluationPriorities.empty() &&
      evaluationPriorities.size() != logicalNetCount)
    return pathFinderError("evaluation-priority vector has the wrong width");

  netOrder_.clear();
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount; ++logicalNet) {
    auto projection = projectLogicalNet(candidate, costs, logicalNet);
    if (!projection)
      return projection.takeError();
    netOrder_.push_back(
        {projection->routeStateRank, projection->conflictPressure,
         evaluationPriorities.empty() ? 0 : evaluationPriorities[logicalNet],
         logicalNet});
  }
  llvm::sort(netOrder_, [](const NetOrderEntry &lhs, const NetOrderEntry &rhs) {
    if (lhs.routeStateRank != rhs.routeStateRank)
      return lhs.routeStateRank < rhs.routeStateRank;
    if (lhs.conflictPressure != rhs.conflictPressure)
      return lhs.conflictPressure > rhs.conflictPressure;
    if (lhs.evaluationPriority != rhs.evaluationPriority)
      return lhs.evaluationPriority > rhs.evaluationPriority;
    return lhs.logicalNet < rhs.logicalNet;
  });
  return llvm::Error::success();
}

llvm::Expected<SpatialPathFinderClosureResult>
SpatialPathFinderRouterScratch::routeToClosure(
    SpatialCandidateState &candidate, SpatialCandidateScratch &candidateScratch,
    SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  if (llvm::Error error = costs.resetFromCandidate())
    return std::move(error);
  auto moveOrError = candidate.beginMove(candidateScratch);
  if (!moveOrError)
    return moveOrError.takeError();
  SpatialMoveTransaction move = std::move(*moveOrError);

  auto result = routeToClosureInMove(move, candidate, costs, limits,
                                     evaluationPriorities);
  if (!result)
    return rollbackIteration(move, costs, result.takeError());
  auto closed = move.close();
  if (!closed)
    return rollbackIteration(move, costs, closed.takeError());
  if (!*closed)
    return rollbackIteration(
        move, costs,
        llvm::make_error<SpatialPathFinderClosureFailure>(
            SpatialPathFinderClosureFailure::Kind::
                SelectedCombinationalHandshakeCycle,
            "Spatial PathFinder selected a combinational handshake cycle"));
  if (llvm::Error error = move.commit())
    return error;
  return result;
}

llvm::Expected<SpatialPathFinderClosureResult>
SpatialPathFinderRouterScratch::routeToClosureInMove(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
    llvm::ArrayRef<RouteCost> evaluationPriorities) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  if (limits.endpointExpansionLimit == 0 || limits.iterationLimit == 0)
    return pathFinderError("routing work limits must be positive");
  if (!evaluationPriorities.empty() &&
      evaluationPriorities.size() !=
          candidate.problem().transfers().logicalNets().size())
    return pathFinderError("evaluation-priority vector has the wrong width");
  if (costs.selectedLogicalNet())
    return pathFinderError("route costs already have a selected logical net");

  const auto allRouted = [&candidate] {
    for (PnrIndex logicalNet = 0;
         logicalNet < candidate.problem().transfers().logicalNets().size();
         ++logicalNet)
      if (candidate.routeTree(logicalNet).isUnrouted())
        return false;
    return true;
  };
  if (allRouted() && !costs.hasCapacityOveruse())
    return SpatialPathFinderClosureResult{};

  for (std::uint64_t iteration = 0; iteration < limits.iterationLimit;
       ++iteration) {
    if (llvm::Error error =
            buildCanonicalNetOrder(candidate, costs, evaluationPriorities))
      return std::move(error);

    constraintSweepNets_.clear();
    for (const NetOrderEntry &entry : netOrder_)
      constraintSweepNets_.push_back(entry.logicalNet);
    if (llvm::Error error =
            netRouter_.beginConstraintSweep(constraintSweepNets_))
      return std::move(error);

    for (const NetOrderEntry &entry : netOrder_) {
      auto projection = projectLogicalNet(candidate, costs, entry.logicalNet);
      if (!projection)
        return projection.takeError();
      if (llvm::Error error =
              costs.selectLogicalNet(entry.logicalNet, activeClaimBits_))
        return std::move(error);
      auto route =
          netRouter_.routeWholeNet(move, candidate, costs, entry.logicalNet,
                                   limits.endpointExpansionLimit);
      if (!route)
        return route.takeError();
      if (llvm::Error error = costs.acceptSelectedLogicalNet())
        return std::move(error);
      if (llvm::Error error = netRouter_.finishConstraintNet(entry.logicalNet))
        return std::move(error);
    }

    const std::uint64_t completedIterations = iteration + 1;
    if (!costs.hasCapacityOveruse())
      return SpatialPathFinderClosureResult{completedIterations};

    if (completedIterations == limits.iterationLimit)
      return llvm::make_error<SpatialPathFinderClosureFailure>(
          SpatialPathFinderClosureFailure::Kind::NonClosure,
          "Spatial PathFinder exhausted its iteration limit before capacity "
          "closure");
    if (llvm::Error error = costs.advancePathFinderIteration())
      return std::move(error);
  }
  llvm_unreachable("positive iteration limit executes or returns");
}

llvm::Expected<RouteCost> SpatialPathFinderRouterScratch::routeWholeNetInMove(
    SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
    SpatialRouteCostState &costs, PnrIndex logicalNet,
    std::uint64_t endpointExpansionLimit) {
  if (!preparedProblem_ || preparedProblem_ != &candidate.problem())
    return pathFinderError("scratch is not prepared for the candidate freeze");
  if (!costs.isBoundTo(candidate))
    return pathFinderError("route costs are bound to another candidate");
  return netRouter_.routeWholeNet(move, candidate, costs, logicalNet,
                                  endpointExpansionLimit);
}

llvm::Error SpatialPathFinderRouterScratch::beginConstraintSweep(
    llvm::ArrayRef<PnrIndex> logicalNets) {
  return netRouter_.beginConstraintSweep(logicalNets);
}

llvm::Error
SpatialPathFinderRouterScratch::finishConstraintNet(PnrIndex logicalNet) {
  return netRouter_.finishConstraintNet(logicalNet);
}

std::size_t SpatialPathFinderRouterScratch::retainedStorageBytes() const {
  return netRouter_.retainedStorageBytes() + retainedBytes(netOrder_) +
         retainedBytes(activeClaimBits_) + retainedBytes(claimEpochs_) +
         retainedBytes(capacityEpochs_) + retainedBytes(capacityNetQCosts_) +
         retainedBytes(touchedCapacities_) +
         retainedBytes(constraintSweepNets_);
}
