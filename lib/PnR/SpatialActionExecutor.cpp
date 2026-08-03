#include "PnR/SpatialActionExecutor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <system_error>
#include <type_traits>
#include <utility>

using namespace loom;
using namespace loom::pnr;

char SpatialActionTransitionFailure::ID;

void SpatialActionTransitionFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SpatialActionTransitionFailure::convertToErrorCode() const {
  return std::make_error_code(
      kind_ == SpatialActionTransitionFailureKind::WorkLimit
          ? std::errc::resource_unavailable_try_again
          : std::errc::invalid_argument);
}

namespace {

llvm::Error executorError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial Action execution: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

bool rangeContains(PnrIndex offset, PnrIndex count, PnrIndex value) {
  return value >= offset && value - offset < count;
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

llvm::Expected<bool> classifyRetainableRouteFailure(llvm::Error failure) {
  bool retainable = false;
  llvm::Error unhandled = llvm::handleErrors(
      std::move(failure),
      [&](const EndpointRouteSearchFailure &routeFailure) -> llvm::Error {
        retainable =
            routeFailure.kind() ==
                EndpointRouteSearchFailureKind::Unreachable ||
            routeFailure.kind() == EndpointRouteSearchFailureKind::WorkLimit;
        if (retainable)
          return llvm::Error::success();
        std::string message;
        llvm::raw_string_ostream stream(message);
        routeFailure.log(stream);
        return llvm::make_error<EndpointRouteSearchFailure>(routeFailure.kind(),
                                                            stream.str());
      });
  if (unhandled)
    return std::move(unhandled);
  return retainable;
}

llvm::Error classifyTransitionFailure(llvm::Error failure) {
  return llvm::handleErrors(
      std::move(failure),
      [&](const EndpointRouteSearchFailure &routeFailure) -> llvm::Error {
        if (routeFailure.kind() == EndpointRouteSearchFailureKind::Invalid ||
            routeFailure.kind() ==
                EndpointRouteSearchFailureKind::ArithmeticOverflow) {
          std::string message;
          llvm::raw_string_ostream stream(message);
          routeFailure.log(stream);
          return llvm::make_error<EndpointRouteSearchFailure>(
              routeFailure.kind(), stream.str());
        }
        std::string message;
        llvm::raw_string_ostream stream(message);
        routeFailure.log(stream);
        return llvm::make_error<SpatialActionTransitionFailure>(
            routeFailure.kind() == EndpointRouteSearchFailureKind::WorkLimit
                ? SpatialActionTransitionFailureKind::WorkLimit
                : SpatialActionTransitionFailureKind::IntrinsicInvalid,
            stream.str());
      },
      [&](const SpatialPathFinderClosureFailure &closureFailure)
          -> llvm::Error {
        std::string message;
        llvm::raw_string_ostream stream(message);
        closureFailure.log(stream);
        return llvm::make_error<SpatialActionTransitionFailure>(
            closureFailure.kind() ==
                    SpatialPathFinderClosureFailure::Kind::NonClosure
                ? SpatialActionTransitionFailureKind::WorkLimit
                : SpatialActionTransitionFailureKind::IntrinsicInvalid,
            stream.str());
      });
}

} // namespace

SpatialActionProbe::SpatialActionProbe(
    SpatialActionExecutorScratch &owner, SpatialMoveTransaction move,
    dse::ObjectiveVector objective,
    dse::ObjectiveSignedDifference energyDifference, bool globalRouting)
    : owner_(&owner), move_(std::move(move)), objective_(std::move(objective)),
      energyDifference_(energyDifference), globalRouting_(globalRouting) {}

SpatialActionProbe::SpatialActionProbe(SpatialActionProbe &&other) noexcept
    : owner_(other.owner_), move_(std::move(other.move_)),
      objective_(std::move(other.objective_)),
      energyDifference_(other.energyDifference_),
      globalRouting_(other.globalRouting_) {
  other.owner_ = nullptr;
}

SpatialActionProbe::~SpatialActionProbe() {
  if (owner_)
    llvm::cantFail(discard());
}

llvm::Error SpatialActionProbe::commit() {
  if (!owner_)
    return executorError("probe is no longer active");
  SpatialActionExecutorScratch *owner = owner_;
  if (llvm::Error error = move_.commit())
    return error;
  owner->currentObjective_ = objective_;
  owner->activeProbe_ = false;
  owner_ = nullptr;
  return llvm::Error::success();
}

llvm::Error SpatialActionProbe::discard() {
  if (!owner_)
    return executorError("probe is no longer active");
  SpatialActionExecutorScratch *owner = owner_;
  move_.rollback();
  llvm::Error synchronization =
      globalRouting_ ? owner->routeCosts_->resetFromCandidate()
                     : owner->routeCosts_->synchronizeCandidateTraversals(
                           owner->routeCostTraversals_);
  owner->activeProbe_ = false;
  owner_ = nullptr;
  return synchronization;
}

llvm::Expected<SpatialActionResolution>
SpatialActionProbe::resolve(std::uint64_t temperature,
                            DeterministicPnrRandomStream &acceptanceStream) {
  if (!owner_)
    return executorError("probe is no longer active");
  auto accepted =
      acceptAnnealingDelta(energyDifference_, temperature, acceptanceStream);
  if (!accepted)
    return accepted.takeError();
  dse::ObjectiveVector resolved =
      *accepted ? objective_ : *owner_->currentObjective_;
  if (llvm::Error error = *accepted ? commit() : discard())
    return std::move(error);
  return SpatialActionResolution{*accepted, std::move(resolved)};
}

llvm::Error
SpatialActionExecutorScratch::prepare(SpatialCandidateState &candidate) {
  if (activeProbe_)
    return executorError("cannot prepare while a probe is active");
  if (llvm::Error error = candidate.verify())
    return error;
  if (llvm::Error error = candidateScratch_.prepare(candidate.problem()))
    return error;
  if (llvm::Error error = router_.prepare(candidate.problem()))
    return error;
  auto routeCosts = SpatialRouteCostState::create(candidate);
  if (!routeCosts)
    return routeCosts.takeError();
  auto objective = candidate.problem().objectiveProgram().evaluate(candidate);
  if (!objective)
    return objective.takeError();

  routeCosts_.emplace(std::move(*routeCosts));
  currentObjective_.emplace(std::move(*objective));
  netMarks_.assign(candidate.problem().transfers().logicalNets().size(), 0);
  affectedNets_.clear();
  affectedNets_.reserve(netMarks_.size());
  routeCostTraversals_.clear();
  routeCostTraversals_.reserve(
      candidate.problem().routing().traversals().size());
  netEpoch_ = 0;
  candidate_ = &candidate;
  return llvm::Error::success();
}

const dse::ObjectiveVector &
SpatialActionExecutorScratch::currentObjective() const {
  assert(currentObjective_ && "Spatial Action executor is not prepared");
  return *currentObjective_;
}

void SpatialActionExecutorScratch::beginDependencyClosure() {
  ++netEpoch_;
  if (netEpoch_ == 0) {
    std::fill(netMarks_.begin(), netMarks_.end(), 0);
    netEpoch_ = 1;
  }
  affectedNets_.clear();
  routeCostTraversals_.clear();
  globalRouting_ = false;
}

llvm::Error SpatialActionExecutorScratch::markNet(PnrIndex logicalNet) {
  if (logicalNet >= netMarks_.size())
    return executorError("Action dependency net is out of range");
  if (netMarks_[logicalNet] == netEpoch_)
    return llvm::Error::success();
  netMarks_[logicalNet] = netEpoch_;
  affectedNets_.push_back(logicalNet);
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::applyComputeBinding(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialComputeBindingAction action) {
  if (action.realization >=
      candidate.problem().realizations().computeRealizations().size())
    return executorError("compute realization is out of range");
  const PnrIndex oldPlacement =
      candidate.computeBinding(action.realization).placement;
  if (llvm::Error error = move.setComputeBinding(
          action.realization, action.placement, action.instructionContext))
    return error;
  if (oldPlacement == action.placement)
    return llvm::Error::success();

  const auto &problem = candidate.problem();
  const auto &ports = problem.ports();
  const auto &realizations = problem.realizations();
  const FrozenSpatialComputeRealization &owner =
      realizations.computeRealizations()[action.realization];
  const auto offsets = ports.computeRealizationDemandOffsets();
  for (PnrIndex demand : ports.computeRealizationDemands().slice(
           offsets[action.realization],
           offsets[action.realization + 1] - offsets[action.realization])) {
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
    const PnrIndex localPlacement = action.placement - owner.placementOffset;
    const FrozenSpatialPortPlacementDomain &domain =
        ports.placementDomains()[record.placementDomainOffset + localPlacement];
    const PnrIndex current = candidate.portAttachment(demand);
    const PnrIndex replacement =
        rangeContains(domain.attachmentOptionOffset,
                      domain.attachmentOptionCount, current)
            ? current
            : domain.attachmentOptionOffset;
    if (llvm::Error error = move.setPortAttachment(demand, replacement))
      return error;
    if (llvm::Error error = markNet(record.logicalNet))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::applyMemoryBinding(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialMemoryBindingAction action) {
  if (action.realization >=
      candidate.problem().realizations().memoryRealizations().size())
    return executorError("memory realization is out of range");
  const PnrIndex oldPlacement =
      candidate.memoryBinding(action.realization).placement;
  if (llvm::Error error =
          move.setMemoryBinding(action.realization, action.placement))
    return error;
  if (oldPlacement == action.placement)
    return llvm::Error::success();

  const auto &problem = candidate.problem();
  const auto &ports = problem.ports();
  const auto &realizations = problem.realizations();
  const FrozenSpatialMemoryRealization &owner =
      realizations.memoryRealizations()[action.realization];
  const auto demandOffsets = ports.memoryRealizationDemandOffsets();
  for (PnrIndex demand : ports.memoryRealizationDemands().slice(
           demandOffsets[action.realization],
           demandOffsets[action.realization + 1] -
               demandOffsets[action.realization])) {
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
    const PnrIndex localPlacement = action.placement - owner.placementOffset;
    const FrozenSpatialPortPlacementDomain &domain =
        ports.placementDomains()[record.placementDomainOffset + localPlacement];
    const PnrIndex current = candidate.portAttachment(demand);
    const PnrIndex replacement =
        rangeContains(domain.attachmentOptionOffset,
                      domain.attachmentOptionCount, current)
            ? current
            : domain.attachmentOptionOffset;
    if (llvm::Error error = move.setPortAttachment(demand, replacement))
      return error;
    if (llvm::Error error = markNet(record.logicalNet))
      return error;
  }

  const auto &handshake = problem.handshake();
  const auto &memory = problem.memory();
  const PnrIndex operationDomainOffset =
      handshake.memoryPlacementDomainOffsets()[action.placement];
  for (PnrIndex localActor = 0; localActor < owner.actorCount; ++localActor) {
    const PnrIndex actor = owner.actorOffset + localActor;
    const FrozenSpatialMemoryOperationHandshakeDomain &planDomain =
        handshake.memoryOperationDomains()[operationDomainOffset + localActor];
    const PnrIndex currentPlan = candidate.memoryOperationPlan(actor);
    const PnrIndex replacementPlan =
        rangeContains(planDomain.planOffset, planDomain.planCount, currentPlan)
            ? currentPlan
            : planDomain.planOffset;
    if (llvm::Error error = move.setMemoryOperationPlan(actor, replacementPlan))
      return error;

    for (PnrIndex use = memory.actorUseOffsets()[actor];
         use < memory.actorUseOffsets()[actor + 1]; ++use) {
      auto dispatchDomain = candidate.memoryDispatchDomain(use);
      if (!dispatchDomain)
        return dispatchDomain.takeError();
      const PnrIndex currentDispatch = candidate.memoryUseDispatches_[use];
      const PnrIndex replacementDispatch =
          rangeContains((*dispatchDomain)->optionOffset,
                        (*dispatchDomain)->optionCount, currentDispatch)
              ? currentDispatch
              : (*dispatchDomain)->optionOffset;
      if (llvm::Error error =
              move.setMemoryUseDispatch(use, replacementDispatch))
        return error;
    }
  }
  return llvm::Error::success();
}

llvm::Error
SpatialActionExecutorScratch::apply(SpatialMoveTransaction &move,
                                    SpatialCandidateState &candidate,
                                    const SpatialMappingAction &action) {
  return std::visit(
      [&](const auto &category) -> llvm::Error {
        using Category = std::decay_t<decltype(category)>;
        if constexpr (std::is_same_v<Category,
                                     SpatialRealizationBindingAction>) {
          return std::visit(
              [&](const auto &choice) -> llvm::Error {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialComputeBindingAction>)
                  return applyComputeBinding(move, candidate, choice);
                else
                  return applyMemoryBinding(move, candidate, choice);
              },
              category);
        } else if constexpr (std::is_same_v<Category,
                                            SpatialTransportRoutingAction>) {
          return std::visit(
              [&](const auto &choice) -> llvm::Error {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialWholeNetRoutingAction>)
                  return markNet(choice.logicalNet);
                else if constexpr (std::is_same_v<Choice,
                                                  SpatialGlobalRoutingAction>) {
                  globalRouting_ = true;
                  return llvm::Error::success();
                } else {
                  return executorError(
                      "routing scope has no production executor");
                }
              },
              category);
        } else {
          return std::visit(
              [&](const auto &choice) -> llvm::Error {
                using Choice = std::decay_t<decltype(choice)>;
                if constexpr (std::is_same_v<Choice,
                                             SpatialPortAttachmentAction>) {
                  if (choice.demand >=
                      candidate.problem().ports().portDemands().size())
                    return executorError("port Action anchor is out of range");
                  if (llvm::Error error = move.setPortAttachment(
                          choice.demand, choice.attachmentOption))
                    return error;
                  return markNet(candidate.problem()
                                     .ports()
                                     .portDemands()[choice.demand]
                                     .logicalNet);
                } else if constexpr (
                    std::is_same_v<Choice,
                                   SpatialGraphBoundaryAttachmentAction>) {
                  if (choice.boundary >=
                      candidate.problem().ports().graphBoundaries().size())
                    return executorError(
                        "graph-boundary Action anchor is out of range");
                  if (llvm::Error error = move.setGraphBoundaryAttachment(
                          choice.boundary, choice.attachmentOption))
                    return error;
                  return markNet(candidate.problem()
                                     .ports()
                                     .graphBoundaries()[choice.boundary]
                                     .logicalNet);
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialMemoryOperationPlanAction>) {
                  return move.setMemoryOperationPlan(choice.actor, choice.plan);
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialLogicalMemoryBindingAction>) {
                  return move.setLogicalMemoryBinding(
                      choice.binding, choice.target,
                      choice.physicalOffsetBytes);
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialMemoryUseDispatchAction>) {
                  return move.setMemoryUseDispatch(choice.use,
                                                   choice.dispatchOption);
                } else {
                  return move.setMemoryExposureSelection(choice.exposure,
                                                         choice.exposureOption);
                }
              },
              category);
        }
      },
      action);
}

llvm::Error SpatialActionExecutorScratch::routeAffectedNets(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  llvm::sort(affectedNets_);
  const bool admitsUnrouted = llvm::is_contained(
      candidate.problem().config().policy().temporaryViolations.admitted,
      ResolvedPnrViolationKind::UnroutedObligation);
  const std::uint64_t endpointLimit =
      candidate.problem()
          .config()
          .policy()
          .search.routing.endpointExpansionLimit;
  for (PnrIndex logicalNet : affectedNets_) {
    if (llvm::Error error = routeCosts_->selectLogicalNet(logicalNet))
      return error;
    auto route = router_.routeWholeNetInMove(move, candidate, *routeCosts_,
                                             logicalNet, endpointLimit);
    if (!route) {
      if (!admitsUnrouted)
        return route.takeError();
      auto retainable = classifyRetainableRouteFailure(route.takeError());
      if (!retainable)
        return retainable.takeError();
      if (!*retainable)
        return executorError("route failure was not retainable");
      if (llvm::Error error = move.ripUpWholeRoute(logicalNet))
        return error;
      if (llvm::Error error = routeCosts_->updateSelectedLogicalNetClaims({}))
        return error;
    }
    if (llvm::Error error = routeCosts_->acceptSelectedLogicalNet())
      return error;
  }
  return llvm::Error::success();
}

llvm::Error
SpatialActionExecutorScratch::restoreAfterFailure(SpatialMoveTransaction &move,
                                                  llvm::Error failure) {
  move.rollback();
  if (llvm::Error reset = routeCosts_->resetFromCandidate())
    return llvm::joinErrors(std::move(failure), std::move(reset));
  return failure;
}

llvm::Expected<SpatialActionProbe>
SpatialActionExecutorScratch::probe(SpatialCandidateState &candidate,
                                    const SpatialMappingAction &action) {
  return probeBatch(candidate,
                    llvm::ArrayRef<SpatialMappingAction>(&action, 1));
}

llvm::Expected<SpatialActionProbe> SpatialActionExecutorScratch::probeBatch(
    SpatialCandidateState &candidate,
    llvm::ArrayRef<SpatialMappingAction> actions) {
  if (activeProbe_)
    return executorError("another Action probe is active");
  if (candidate_ != &candidate || !routeCosts_ || !currentObjective_)
    return executorError("executor was not prepared for this candidate");
  if (llvm::Error error = validateCanonicalSpatialActionBatch(actions))
    return std::move(error);
  beginDependencyClosure();

  auto moveOrError = candidate.beginMove(candidateScratch_);
  if (!moveOrError)
    return moveOrError.takeError();
  SpatialMoveTransaction move = std::move(*moveOrError);
  for (const SpatialMappingAction &action : actions)
    if (llvm::Error error = apply(move, candidate, action))
      return restoreAfterFailure(move, std::move(error));

  if (globalRouting_) {
    const auto &routing = candidate.problem().config().policy().search.routing;
    auto closure = router_.routeToClosureInMove(
        move, candidate, *routeCosts_,
        {routing.endpointExpansionLimit, routing.negotiationIterationLimit},
        {});
    if (!closure)
      return restoreAfterFailure(
          move, classifyTransitionFailure(closure.takeError()));
  } else if (llvm::Error error = routeAffectedNets(move, candidate)) {
    return restoreAfterFailure(move,
                               classifyTransitionFailure(std::move(error)));
  }

  auto closed = move.close();
  if (!closed)
    return restoreAfterFailure(move, closed.takeError());
  if (!*closed)
    return restoreAfterFailure(
        move, llvm::make_error<SpatialActionTransitionFailure>(
                  SpatialActionTransitionFailureKind::IntrinsicInvalid,
                  "Spatial Action selected a combinational handshake cycle"));
  routeCostTraversals_.assign(move.touchedRouteTraversals().begin(),
                              move.touchedRouteTraversals().end());

  auto objective = candidate.problem().objectiveProgram().evaluate(candidate);
  if (!objective)
    return restoreAfterFailure(move, objective.takeError());
  auto difference =
      candidate.problem().objectiveProgram().selectedEnergyDifference(
          *objective, *currentObjective_);
  if (!difference)
    return restoreAfterFailure(move, difference.takeError());

  activeProbe_ = true;
  return SpatialActionProbe(*this, std::move(move), std::move(*objective),
                            *difference, globalRouting_);
}

std::size_t SpatialActionExecutorScratch::retainedStorageBytes() const {
  return candidateScratch_.retainedStorageBytes() +
         router_.retainedStorageBytes() +
         (routeCosts_ ? routeCosts_->retainedStorageBytes() : 0) +
         retainedBytes(netMarks_) + retainedBytes(affectedNets_) +
         retainedBytes(routeCostTraversals_);
}
