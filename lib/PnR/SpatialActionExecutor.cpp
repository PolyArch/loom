#include "PnR/SpatialActionExecutor.h"

#include "InitializerRelationSolver.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialMemoryCompatibility.h"
#include "SpatialMemoryConstraintModel.h"
#include "SpatialRouteConstraintModel.h"

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

llvm::Error intrinsicTransitionFailure(const llvm::Twine &message) {
  return llvm::make_error<SpatialActionTransitionFailure>(
      SpatialActionTransitionFailureKind::IntrinsicInvalid, message.str());
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

llvm::Error classifyTransitionFailure(llvm::Error failure,
                                      SpatialActionExecutionContext context) {
  return llvm::handleErrors(
      std::move(failure),
      [&](const EndpointRouteSearchFailure &routeFailure) -> llvm::Error {
        if (routeFailure.kind() == EndpointRouteSearchFailureKind::Invalid) {
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
        if (closureFailure.kind() ==
                SpatialPathFinderClosureFailure::Kind::FixedTerminalCapacityCut &&
            context == SpatialActionExecutionContext::FinalClosure)
          return llvm::make_error<SpatialPathFinderClosureFailure>(
              closureFailure.kind(), stream.str(),
              closureFailure.certificateCapacity(),
              closureFailure.mandatoryUsage(),
              closureFailure.physicalCapacity(),
              std::vector<PnrIndex>(closureFailure.forcedLogicalNets().begin(),
                                    closureFailure.forcedLogicalNets().end()));
        return llvm::make_error<SpatialActionTransitionFailure>(
            closureFailure.kind() ==
                        SpatialPathFinderClosureFailure::Kind::NonClosure ||
                    closureFailure.kind() ==
                        SpatialPathFinderClosureFailure::Kind::NoProgress
                ? SpatialActionTransitionFailureKind::WorkLimit
                : SpatialActionTransitionFailureKind::IntrinsicInvalid,
            stream.str());
      },
      [&](const RoutingNegotiationError &negotiationError) -> llvm::Error {
        std::string message;
        llvm::raw_string_ostream stream(message);
        negotiationError.log(stream);
        if (negotiationError.kind() ==
            RoutingNegotiationError::Kind::InvalidPolicy)
          return llvm::make_error<RoutingNegotiationError>(
              negotiationError.kind(), stream.str());
        return llvm::make_error<SpatialActionTransitionFailure>(
            SpatialActionTransitionFailureKind::IntrinsicInvalid, stream.str());
      });
}

llvm::Error classifyRelationFailure(llvm::Error failure) {
  return llvm::handleErrors(
      std::move(failure),
      [&](const detail::InitializerRelationSolveFailure &relationFailure)
          -> llvm::Error {
        std::string message;
        llvm::raw_string_ostream stream(message);
        relationFailure.log(stream);
        return llvm::make_error<SpatialActionTransitionFailure>(
            relationFailure.kind() ==
                    detail::InitializerRelationSolveFailureKind::WorkLimit
                ? SpatialActionTransitionFailureKind::WorkLimit
                : SpatialActionTransitionFailureKind::IntrinsicInvalid,
            stream.str());
      });
}

} // namespace

SpatialActionExecutorScratch::SpatialActionExecutorScratch() = default;
SpatialActionExecutorScratch::~SpatialActionExecutorScratch() = default;

SpatialActionProbe::SpatialActionProbe(
    SpatialActionExecutorScratch &owner, SpatialMoveTransaction move,
    dse::ObjectiveVector objective,
    dse::ObjectiveSignedDifference energyDifference, bool globalRouting,
    bool semanticChange)
    : owner_(&owner), move_(std::move(move)), objective_(std::move(objective)),
      energyDifference_(energyDifference), globalRouting_(globalRouting),
      semanticChange_(semanticChange) {}

SpatialActionProbe::SpatialActionProbe(SpatialActionProbe &&other) noexcept
    : owner_(other.owner_), move_(std::move(other.move_)),
      objective_(std::move(other.objective_)),
      energyDifference_(other.energyDifference_),
      globalRouting_(other.globalRouting_),
      semanticChange_(other.semanticChange_) {
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
  if (!semanticChange_) {
    if (llvm::Error error = discard())
      return std::move(error);
    return SpatialActionResolution{false, std::move(objective_)};
  }
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
  const detail::SpatialBindingRelationModel &bindings =
      candidate.problem().bindingRelations();
  relationSolver_ =
      std::make_unique<detail::InitializerRelationSolver>(bindings.relations());
  if (!memoryConstraintScratch_)
    memoryConstraintScratch_ =
        std::make_unique<detail::SpatialMemoryConstraintScratch>();
  if (llvm::Error error =
          candidate.problem().memoryConstraints().prepareScratch(
              *memoryConstraintScratch_))
    return error;
  fixedRelationChoices_.resize(bindings.decisionCount());
  relationDecisionMarks_.resize(bindings.decisionCount());
  explicitAttachmentMarks_.resize(bindings.decisionCount());
  relationDecisionQueue_.clear();
  relationDecisionQueue_.reserve(bindings.decisionCount());
  changedBindingRoots_.clear();
  changedBindingRoots_.reserve(bindings.realizationDecisionCount());
  const std::size_t logicalMemoryCount =
      candidate.problem().memory().logicalBindings().size();
  explicitLogicalMemorySelections_.resize(logicalMemoryCount);
  explicitLogicalMemoryMarks_.resize(logicalMemoryCount);
  explicitLogicalMemoryBindings_.clear();
  explicitLogicalMemoryBindings_.reserve(logicalMemoryCount);
  explicitLogicalMemoryChoices_.clear();
  explicitLogicalMemoryChoices_.reserve(logicalMemoryCount);
  changedLogicalMemoryMarks_.resize(logicalMemoryCount);
  changedLogicalMemoryBindings_.clear();
  changedLogicalMemoryBindings_.reserve(logicalMemoryCount);
  const std::size_t memoryGroupCount =
      candidate.problem().memory().serviceUseGroups().size();
  explicitMemoryDispatchPatterns_.resize(memoryGroupCount);
  explicitMemoryDispatchGroupMarks_.resize(memoryGroupCount);
  explicitMemoryDispatchGroups_.clear();
  explicitMemoryDispatchGroups_.reserve(memoryGroupCount);
  const std::size_t memoryUseCount =
      candidate.problem().memory().rootedUses().size();
  explicitMemoryDispatchSelections_.resize(memoryUseCount);
  explicitMemoryDispatchUseMarks_.resize(memoryUseCount);
  const std::size_t memoryExposureCount =
      candidate.problem().memory().exposures().size();
  explicitMemoryExposureSelections_.resize(memoryExposureCount);
  explicitMemoryExposureMarks_.resize(memoryExposureCount);
  currentObjective_.emplace(std::move(*objective));
  netMarks_.assign(candidate.problem().transfers().logicalNets().size(), 0);
  pendingRouteKinds_.resize(netMarks_.size());
  pendingRouteAnchors_.resize(netMarks_.size());
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
  std::fill(relationDecisionMarks_.begin(), relationDecisionMarks_.end(), 0);
  std::fill(explicitAttachmentMarks_.begin(), explicitAttachmentMarks_.end(),
            0);
  relationDecisionQueue_.clear();
  changedBindingRoots_.clear();
  std::fill(explicitLogicalMemoryMarks_.begin(),
            explicitLogicalMemoryMarks_.end(), 0);
  explicitLogicalMemoryBindings_.clear();
  explicitLogicalMemoryChoices_.clear();
  std::fill(changedLogicalMemoryMarks_.begin(),
            changedLogicalMemoryMarks_.end(), 0);
  changedLogicalMemoryBindings_.clear();
  std::fill(explicitMemoryDispatchGroupMarks_.begin(),
            explicitMemoryDispatchGroupMarks_.end(), 0);
  explicitMemoryDispatchGroups_.clear();
  std::fill(explicitMemoryDispatchUseMarks_.begin(),
            explicitMemoryDispatchUseMarks_.end(), 0);
  std::fill(explicitMemoryExposureMarks_.begin(),
            explicitMemoryExposureMarks_.end(), 0);
  globalRouting_ = false;
}

void SpatialActionExecutorScratch::markChangedBindingRoot(PnrIndex decision) {
  if (decision >= relationDecisionMarks_.size() ||
      relationDecisionMarks_[decision])
    return;
  relationDecisionMarks_[decision] = 1;
  changedBindingRoots_.push_back(decision);
}

void SpatialActionExecutorScratch::markExplicitAttachment(PnrIndex decision) {
  if (decision < explicitAttachmentMarks_.size())
    explicitAttachmentMarks_[decision] = 1;
}

llvm::Error SpatialActionExecutorScratch::markNet(PnrIndex logicalNet) {
  if (logicalNet >= netMarks_.size())
    return executorError("Action dependency net is out of range");
  if (globalRouting_)
    return llvm::Error::success();
  for (PnrIndex member :
       candidate_->problem().routeConstraints().equalityClosure(logicalNet)) {
    if (member >= netMarks_.size())
      return executorError("route equality closure net is out of range");
    if (netMarks_[member] == netEpoch_) {
      pendingRouteKinds_[member] = PendingRouteKind::WholeNet;
      pendingRouteAnchors_[member] = getInvalidPnrIndex();
      continue;
    }
    netMarks_[member] = netEpoch_;
    pendingRouteKinds_[member] = PendingRouteKind::WholeNet;
    pendingRouteAnchors_[member] = getInvalidPnrIndex();
    affectedNets_.push_back(member);
  }
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::markLocalNet(PnrIndex logicalNet,
                                                       PendingRouteKind kind,
                                                       PnrIndex localAnchor) {
  if (logicalNet >= netMarks_.size())
    return executorError("local routing net is out of range");
  if (netMarks_[logicalNet] == netEpoch_)
    return executorError("local routing scope overlaps another dependency");
  if (llvm::Error error = markNet(logicalNet))
    return error;
  pendingRouteKinds_[logicalNet] = kind;
  pendingRouteAnchors_[logicalNet] = localAnchor;
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::markWitnessRegion(
    SpatialWitnessRegionRoutingAction action) {
  const FrozenSpatialPnrProblem &problem = candidate_->problem();
  const auto &transfers = problem.transfers();
  switch (action.witnessKind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
      if (!rangeContains(net.sinkOffset, net.sinkCount, action.witnessOrdinal))
        continue;
      if (candidate_->routeTree(logicalNet).isRouted())
        return executorError("unrouted witness is no longer live");
      return markNet(logicalNet);
    }
    return executorError("unrouted witness ordinal is out of range");
  case ResolvedPnrViolationKind::CapacityOveruse: {
    const auto &routing = problem.routing();
    const PnrIndex capacityCount =
        static_cast<PnrIndex>(problem.resources().capacityDimensions().size());
    if (action.witnessOrdinal >= capacityCount) {
      const PnrIndex domain = action.witnessOrdinal - capacityCount;
      if (domain >= routing.tagContinuity().matchDomains().size() ||
          candidate_->tagDomainResidentCapacityOveruse(domain) == 0)
        return executorError("tag-table capacity witness is no longer live");
      bool marked = false;
      for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
           ++logicalNet) {
        const auto values = candidate_->tagValues(logicalNet);
        for (PnrIndex segment = 0; segment < values.size(); ++segment) {
          if (!llvm::is_contained(
                  candidate_->tagSegmentDomains(logicalNet, segment), domain))
            continue;
          if (llvm::Error error = markNet(logicalNet))
            return error;
          marked = true;
          break;
        }
      }
      if (!marked)
        return executorError("tag-table capacity witness has no selected net");
      return llvm::Error::success();
    }
    const PnrIndex capacity = action.witnessOrdinal;
    if (candidate_->routeCapacityOveruseRaw(capacity) == 0)
      return executorError("route-capacity witness is no longer live");
    const auto claims = routing.capacityRouteClaims().slice(
        routing.capacityRouteClaimOffsets()[capacity],
        routing.capacityRouteClaimOffsets()[capacity + 1] -
            routing.capacityRouteClaimOffsets()[capacity]);
    bool marked = false;
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      if (!llvm::any_of(claims, [&](PnrIndex claim) {
            return candidate_->logicalNetRouteClaimRefcount(logicalNet,
                                                            claim) != 0;
          }))
        continue;
      if (llvm::Error error = markNet(logicalNet))
        return error;
      marked = true;
    }
    if (!marked)
      return executorError("route-capacity witness has no selected net");
    return llvm::Error::success();
  }
  case ResolvedPnrViolationKind::TagUnassigned: {
    PnrIndex ordinal = 0;
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet)
      for (const auto &value : candidate_->tagValues(logicalNet)) {
        if (ordinal == action.witnessOrdinal) {
          if (value)
            return executorError("unassigned-tag witness is no longer live");
          return markNet(logicalNet);
        }
        if (ordinal == getInvalidPnrIndex() - PnrIndex{1})
          return executorError("tag witness ordinal overflows PnrIndex");
        ++ordinal;
      }
    return executorError("unassigned-tag witness ordinal is out of range");
  }
  case ResolvedPnrViolationKind::TagConflict: {
    const PnrIndex domain = action.witnessOrdinal;
    if (domain >= problem.routing().tagContinuity().matchDomains().size() ||
        candidate_->tagDomainConflictCount(domain) == 0)
      return executorError("tag-conflict witness is no longer live");
    bool marked = false;
    for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
         ++logicalNet) {
      const auto values = candidate_->tagValues(logicalNet);
      for (PnrIndex segment = 0; segment < values.size(); ++segment) {
        if (!values[segment] ||
            !llvm::is_contained(
                candidate_->tagSegmentDomains(logicalNet, segment), domain) ||
            !candidate_->tagDomainValueConflicts(domain, *values[segment]))
          continue;
        if (llvm::Error error = markNet(logicalNet))
          return error;
        marked = true;
        break;
      }
    }
    if (!marked)
      return executorError("tag-conflict witness has no selected net");
    return llvm::Error::success();
  }
  case ResolvedPnrViolationKind::HardProgressViolation:
    return executorError(
        "HardProgressViolation has no transport-routing dependency closure");
  }
  llvm_unreachable("unknown Spatial violation kind");
}

llvm::Error SpatialActionExecutorScratch::applyComputeBinding(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    SpatialComputeBindingAction action) {
  if (action.realization >=
      candidate.problem().realizations().computeRealizations().size())
    return executorError("compute realization is out of range");
  const PnrIndex oldPlacement =
      candidate.computeBinding(action.realization).placement;
  const PnrIndex oldContext =
      candidate.computeBinding(action.realization).instructionContext;
  if (llvm::Error error = move.setComputeBinding(
          action.realization, action.placement, action.instructionContext))
    return error;
  if (oldPlacement != action.placement ||
      oldContext != action.instructionContext)
    markChangedBindingRoot(action.realization);
  if (oldPlacement == action.placement)
    return llvm::Error::success();

  const auto &problem = candidate.problem();
  const auto &ports = problem.ports();
  const auto offsets = ports.computeRealizationDemandOffsets();
  for (PnrIndex demand : ports.computeRealizationDemands().slice(
           offsets[action.realization],
           offsets[action.realization + 1] - offsets[action.realization])) {
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
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
  if (oldPlacement != action.placement)
    markChangedBindingRoot(
        candidate.problem().bindingRelations().computeDecisionCount() +
        action.realization);
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
      if (memory.rootedUses()[use].logicalBinding)
        markChangedLogicalMemoryBinding(
            *memory.rootedUses()[use].logicalBinding);
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
                else if constexpr (std::is_same_v<
                                       Choice, SpatialSingleSinkRoutingAction>)
                  return markLocalNet(choice.logicalNet,
                                      PendingRouteKind::SingleSink,
                                      choice.sinkObligation);
                else if constexpr (std::is_same_v<
                                       Choice,
                                       SpatialRootedSubtreeRoutingAction>)
                  return markLocalNet(choice.logicalNet,
                                      PendingRouteKind::RootedSubtree,
                                      choice.rootEndpoint);
                else if constexpr (std::is_same_v<
                                       Choice,
                                       SpatialWitnessRegionRoutingAction>)
                  return markWitnessRegion(choice);
                else if constexpr (std::is_same_v<Choice,
                                                  SpatialGlobalRoutingAction>) {
                  if (globalRouting_)
                    return executorError("Global routing Action is duplicated");
                  globalRouting_ = true;
                  affectedNets_.clear();
                  return llvm::Error::success();
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
                  markExplicitAttachment(candidate.problem()
                                             .bindingRelations()
                                             .portDecisionOffset() +
                                         choice.demand);
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
                  markExplicitAttachment(candidate.problem()
                                             .bindingRelations()
                                             .graphBoundaryDecisionOffset() +
                                         choice.boundary);
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
                  return recordExplicitLogicalMemoryBinding(candidate, choice);
                } else if constexpr (std::is_same_v<
                                         Choice,
                                         SpatialMemoryUseDispatchAction>) {
                  if (llvm::Error error = move.setMemoryUseDispatch(
                          choice.use, choice.dispatchOption))
                    return error;
                  return recordExplicitMemoryDispatch(candidate, choice.use,
                                                      choice.dispatchOption);
                } else {
                  if (llvm::Error error = move.setMemoryExposureSelection(
                          choice.exposure, choice.exposureOption))
                    return error;
                  return recordExplicitMemoryExposure(
                      candidate, choice.exposure, choice.exposureOption);
                }
              },
              category);
        }
      },
      action);
}

llvm::Error SpatialActionExecutorScratch::recordExplicitLogicalMemoryBinding(
    const SpatialCandidateState &candidate,
    SpatialLogicalMemoryBindingAction action) {
  if (action.binding >= explicitLogicalMemorySelections_.size() ||
      action.target >= candidate.problem().memory().bindingTargets().size())
    return executorError("logical-memory Action is out of range");
  const SpatialLogicalMemoryBindingSelection selection{
      action.target, action.physicalOffsetBytes};
  if (explicitLogicalMemoryMarks_[action.binding]) {
    const auto &prior = explicitLogicalMemorySelections_[action.binding];
    if (prior.target != selection.target ||
        prior.physicalOffsetBytes != selection.physicalOffsetBytes)
      return intrinsicTransitionFailure(
          "one ActionBatch selects conflicting logical-memory bindings");
    return llvm::Error::success();
  }
  explicitLogicalMemoryMarks_[action.binding] = 1;
  explicitLogicalMemorySelections_[action.binding] = selection;
  explicitLogicalMemoryBindings_.push_back(action.binding);
  explicitLogicalMemoryChoices_.push_back(selection);
  return llvm::Error::success();
}

llvm::Expected<bool>
SpatialActionExecutorScratch::explicitLogicalMemoryTargetSupported(
    const SpatialCandidateState &candidate, PnrIndex binding,
    PnrIndex targetOrdinal) const {
  auto supported =
      candidate.logicalMemoryBindingTargetSupported(binding, targetOrdinal);
  if (!supported)
    return supported.takeError();
  if (!*supported)
    return false;

  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  if (binding >= memory.logicalBindings().size() ||
      targetOrdinal >= memory.bindingTargets().size())
    return executorError("logical-memory target support is out of range");
  const FrozenSpatialMemoryBindingTargetOption &target =
      memory.bindingTargets()[targetOrdinal];
  const auto patterns =
      candidate.problem().capacity().memoryDispatchOptionPatterns();
  const auto optionSupported = [&](PnrIndex use,
                                   PnrIndex option) -> llvm::Expected<bool> {
    auto domain = candidate.memoryDispatchDomain(use);
    if (!domain)
      return domain.takeError();
    if (!rangeContains((*domain)->optionOffset, (*domain)->optionCount, option))
      return false;
    return detail::memoryDispatchMatchesTarget(
        memory, memory.dispatchOptions()[option], target);
  };

  const auto uses =
      memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                 memory.bindingUseOffsets()[binding + 1] -
                                     memory.bindingUseOffsets()[binding]);
  for (PnrIndex use : uses) {
    if (use >= explicitMemoryDispatchUseMarks_.size())
      return executorError("logical-memory use is out of range");
    if (explicitMemoryDispatchUseMarks_[use]) {
      auto exact = optionSupported(use, explicitMemoryDispatchSelections_[use]);
      if (!exact)
        return exact.takeError();
      if (!*exact)
        return false;
    }
    const PnrIndex group = memory.rootedUseServiceGroups()[use];
    if (group == getInvalidPnrIndex())
      continue;
    if (group >= memory.serviceUseGroups().size() ||
        group >= explicitMemoryDispatchGroupMarks_.size())
      return executorError("logical-memory use group is out of range");
    if (!explicitMemoryDispatchGroupMarks_[group])
      continue;
    const PnrIndex requiredPattern = explicitMemoryDispatchPatterns_[group];
    const FrozenSpatialMemoryServiceUseGroup &record =
        memory.serviceUseGroups()[group];
    if (record.logicalBinding != binding)
      return executorError("logical-memory use group has a foreign binding");
    for (PnrIndex member :
         memory.serviceGroupUses().slice(record.useOffset, record.useCount)) {
      if (member >= explicitMemoryDispatchUseMarks_.size())
        return executorError("logical-memory use group has a foreign member");
      if (explicitMemoryDispatchUseMarks_[member]) {
        const PnrIndex option = explicitMemoryDispatchSelections_[member];
        auto exact = optionSupported(member, option);
        if (!exact)
          return exact.takeError();
        if (!*exact || patterns[option] != requiredPattern)
          return false;
        continue;
      }
      auto domain = candidate.memoryDispatchDomain(member);
      if (!domain)
        return domain.takeError();
      bool matching = false;
      for (PnrIndex option = (*domain)->optionOffset;
           option < (*domain)->optionOffset + (*domain)->optionCount; ++option)
        matching |= patterns[option] == requiredPattern &&
                    detail::memoryDispatchMatchesTarget(
                        memory, memory.dispatchOptions()[option], target);
      if (!matching)
        return false;
    }
  }

  const auto exposures = memory.bindingExposures().slice(
      memory.bindingExposureOffsets()[binding],
      memory.bindingExposureOffsets()[binding + 1] -
          memory.bindingExposureOffsets()[binding]);
  for (PnrIndex exposure : exposures) {
    if (exposure >= explicitMemoryExposureMarks_.size())
      return executorError("logical-memory exposure is out of range");
    if (!explicitMemoryExposureMarks_[exposure])
      continue;
    const PnrIndex option = explicitMemoryExposureSelections_[exposure];
    if (option >= memory.exposureOptions().size() ||
        !detail::memoryExposureMatchesTarget(target,
                                             memory.exposureOptions()[option]))
      return false;
  }
  return true;
}

llvm::Error
SpatialActionExecutorScratch::reconcileExplicitLogicalMemoryBindings(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  if (!explicitLogicalMemoryBindings_.empty()) {
    auto solved = candidate.problem().memoryConstraints().solveCanonicalClosure(
        candidate.logicalMemoryBindings_, explicitLogicalMemoryBindings_,
        explicitLogicalMemoryChoices_,
        candidate.problem()
            .config()
            .policy()
            .search.initializer.assignmentAttemptLimitPerSeed,
        [&](PnrIndex binding, PnrIndex target) -> llvm::Expected<bool> {
          return explicitLogicalMemoryTargetSupported(candidate, binding,
                                                      target);
        },
        *memoryConstraintScratch_);
    if (!solved)
      return llvm::handleErrors(
          solved.takeError(),
          [&](const detail::SpatialMemoryConstraintSolveFailure &)
              -> llvm::Error {
            return llvm::make_error<SpatialActionTransitionFailure>(
                SpatialActionTransitionFailureKind::WorkLimit,
                "Spatial memory relation closure exhausted its assignment "
                "work limit");
          });
    if (!*solved)
      return intrinsicTransitionFailure(
          "logical-memory Action has no relation-closed assignment");

    const auto solution = memoryConstraintScratch_->solution();
    std::optional<PnrIndex> boundaryTarget;
    for (auto [ordinal, target] :
         llvm::enumerate(candidate.problem().memory().bindingTargets()))
      if (std::holds_alternative<FrozenSpatialMemoryBoundaryProxy>(
              target.target)) {
        boundaryTarget = static_cast<PnrIndex>(ordinal);
        break;
      }
    for (PnrIndex binding = 0; binding < solution.size(); ++binding) {
      const auto &current = candidate.logicalMemoryBinding(binding);
      const auto &replacement = solution[binding];
      if (current.target == replacement.target &&
          current.physicalOffsetBytes == replacement.physicalOffsetBytes)
        continue;
      markChangedLogicalMemoryBinding(binding);
      if (!boundaryTarget)
        return executorError("logical-memory closure has no BoundaryProxy");
      if (!std::holds_alternative<FrozenSpatialMemoryBoundaryProxy>(
              candidate.problem()
                  .memory()
                  .bindingTargets()[current.target]
                  .target))
        if (llvm::Error error =
                move.setLogicalMemoryBinding(binding, *boundaryTarget, 0))
          return error;
    }
    for (PnrIndex binding = 0; binding < solution.size(); ++binding) {
      const auto &replacement = solution[binding];
      const auto &current = candidate.logicalMemoryBinding(binding);
      if (current.target == replacement.target &&
          current.physicalOffsetBytes == replacement.physicalOffsetBytes)
        continue;
      if (llvm::Error error = move.setLogicalMemoryBinding(
              binding, replacement.target, replacement.physicalOffsetBytes))
        return error;
    }
    for (PnrIndex binding = 0; binding < solution.size(); ++binding) {
      const auto &replacement = solution[binding];
      const auto &current = candidate.logicalMemoryBinding(binding);
      if (current.target != replacement.target ||
          current.physicalOffsetBytes != replacement.physicalOffsetBytes)
        return executorError(
            "logical-memory closure lost its selected binding");
    }
    for (PnrIndex binding : explicitLogicalMemoryBindings_)
      if (candidate.logicalMemoryBinding(binding).target !=
              explicitLogicalMemorySelections_[binding].target ||
          candidate.logicalMemoryBinding(binding).physicalOffsetBytes !=
              explicitLogicalMemorySelections_[binding].physicalOffsetBytes)
        return executorError(
            "logical-memory closure replaced an explicit choice");
  }
  for (PnrIndex binding : changedLogicalMemoryBindings_)
    if (llvm::Error error =
            reconcileLogicalMemoryBinding(move, candidate, binding))
      return error;
  return llvm::Error::success();
}

void SpatialActionExecutorScratch::markChangedLogicalMemoryBinding(
    PnrIndex binding) {
  if (changedLogicalMemoryMarks_[binding])
    return;
  changedLogicalMemoryMarks_[binding] = 1;
  changedLogicalMemoryBindings_.push_back(binding);
}

llvm::Error SpatialActionExecutorScratch::reconcileLogicalMemoryBinding(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    PnrIndex binding) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  if (binding >= memory.logicalBindings().size())
    return executorError("logical-memory Action anchor is out of range");
  const PnrIndex targetOrdinal = candidate.logicalMemoryBinding(binding).target;
  if (targetOrdinal >= memory.bindingTargets().size())
    return executorError("logical-memory Action selected a foreign target");
  const FrozenSpatialMemoryBindingTargetOption &target =
      memory.bindingTargets()[targetOrdinal];
  const auto patterns =
      candidate.problem().capacity().memoryDispatchOptionPatterns();

  const auto matchingOption = [&](PnrIndex use,
                                  std::optional<PnrIndex> requiredPattern)
      -> llvm::Expected<std::optional<PnrIndex>> {
    auto domain = candidate.memoryDispatchDomain(use);
    if (!domain)
      return domain.takeError();
    for (PnrIndex option = (*domain)->optionOffset;
         option < (*domain)->optionOffset + (*domain)->optionCount; ++option) {
      if (!detail::memoryDispatchMatchesTarget(
              memory, memory.dispatchOptions()[option], target))
        continue;
      if (!requiredPattern || patterns[option] == *requiredPattern)
        return std::optional<PnrIndex>{option};
    }
    return std::optional<PnrIndex>{};
  };
  const auto selectionMatches = [&](PnrIndex use, PnrIndex option) {
    return detail::memoryDispatchMatchesTarget(
        memory, memory.dispatchOptions()[option], target);
  };

  const auto bindingUses =
      memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                 memory.bindingUseOffsets()[binding + 1] -
                                     memory.bindingUseOffsets()[binding]);
  for (PnrIndex use : bindingUses) {
    const PnrIndex group = memory.rootedUseServiceGroups()[use];
    if (group == getInvalidPnrIndex()) {
      const PnrIndex current = candidate.memoryUseDispatch(use);
      if (selectionMatches(use, current))
        continue;
      auto replacement = matchingOption(use, std::nullopt);
      if (!replacement)
        return replacement.takeError();
      if (!*replacement)
        return intrinsicTransitionFailure(
            "logical-memory target has no compatible dispatch");
      if (llvm::Error error = move.setMemoryUseDispatch(use, **replacement))
        return error;
      continue;
    }
    if (group >= memory.serviceUseGroups().size())
      return executorError("memory use selects a foreign service-use group");
    const FrozenSpatialMemoryServiceUseGroup &record =
        memory.serviceUseGroups()[group];
    const auto groupUses =
        memory.serviceGroupUses().slice(record.useOffset, record.useCount);
    if (groupUses.empty() || groupUses.front() != use)
      continue;

    const PnrIndex currentPattern =
        patterns[candidate.memoryUseDispatch(groupUses.front())];
    const bool currentCompatible =
        llvm::all_of(groupUses, [&](PnrIndex member) {
          const PnrIndex option = candidate.memoryUseDispatch(member);
          return selectionMatches(member, option) &&
                 patterns[option] == currentPattern;
        });
    if (currentCompatible)
      continue;

    auto firstDomain = candidate.memoryDispatchDomain(groupUses.front());
    if (!firstDomain)
      return firstDomain.takeError();
    std::optional<PnrIndex> selectedPattern;
    for (PnrIndex option = (*firstDomain)->optionOffset;
         option < (*firstDomain)->optionOffset + (*firstDomain)->optionCount;
         ++option) {
      if (!selectionMatches(groupUses.front(), option))
        continue;
      const PnrIndex pattern = patterns[option];
      bool common = true;
      for (PnrIndex member : groupUses) {
        auto compatible = matchingOption(member, pattern);
        if (!compatible)
          return compatible.takeError();
        if (!*compatible) {
          common = false;
          break;
        }
      }
      if (common) {
        selectedPattern = pattern;
        break;
      }
    }
    if (!selectedPattern)
      return intrinsicTransitionFailure(
          "logical-memory target has no common service UsePattern");
    for (PnrIndex member : groupUses) {
      auto replacement = matchingOption(member, selectedPattern);
      if (!replacement)
        return replacement.takeError();
      if (!*replacement)
        return executorError("common memory dispatch disappeared");
      if (llvm::Error error = move.setMemoryUseDispatch(member, **replacement))
        return error;
    }
  }

  const auto exposures = memory.bindingExposures().slice(
      memory.bindingExposureOffsets()[binding],
      memory.bindingExposureOffsets()[binding + 1] -
          memory.bindingExposureOffsets()[binding]);
  for (PnrIndex exposure : exposures) {
    const PnrIndex current = candidate.memoryExposureSelection(exposure);
    if (detail::memoryExposureMatchesTarget(target,
                                            memory.exposureOptions()[current]))
      continue;
    std::optional<PnrIndex> replacement;
    for (PnrIndex option = 0; option < memory.exposureOptions().size();
         ++option)
      if (detail::memoryExposureMatchesTarget(
              target, memory.exposureOptions()[option])) {
        replacement = option;
        break;
      }
    if (!replacement)
      return intrinsicTransitionFailure(
          "logical-memory target has no compatible exposure");
    if (llvm::Error error =
            move.setMemoryExposureSelection(exposure, *replacement))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::recordExplicitMemoryDispatch(
    const SpatialCandidateState &candidate, PnrIndex use, PnrIndex option) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  if (use >= memory.rootedUseServiceGroups().size() ||
      option >=
          candidate.problem().capacity().memoryDispatchOptionPatterns().size())
    return executorError("memory-dispatch Action is out of range");
  if (use >= explicitMemoryDispatchSelections_.size())
    return executorError("memory-dispatch Action has a foreign use");
  if (explicitMemoryDispatchUseMarks_[use] &&
      explicitMemoryDispatchSelections_[use] != option)
    return intrinsicTransitionFailure(
        "one ActionBatch selects conflicting options for one memory use");
  explicitMemoryDispatchUseMarks_[use] = 1;
  explicitMemoryDispatchSelections_[use] = option;
  const PnrIndex group = memory.rootedUseServiceGroups()[use];
  if (group == getInvalidPnrIndex())
    return llvm::Error::success();
  if (group >= explicitMemoryDispatchPatterns_.size())
    return executorError("memory-dispatch Action has a foreign group");
  const PnrIndex pattern =
      candidate.problem().capacity().memoryDispatchOptionPatterns()[option];
  if (explicitMemoryDispatchGroupMarks_[group]) {
    if (explicitMemoryDispatchPatterns_[group] != pattern)
      return intrinsicTransitionFailure(
          "one ActionBatch selects conflicting memory UsePatterns");
    return llvm::Error::success();
  }
  explicitMemoryDispatchGroupMarks_[group] = 1;
  explicitMemoryDispatchPatterns_[group] = pattern;
  explicitMemoryDispatchGroups_.push_back(group);
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::reconcileExplicitMemoryDispatches(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  const auto patterns =
      candidate.problem().capacity().memoryDispatchOptionPatterns();
  for (PnrIndex group : explicitMemoryDispatchGroups_) {
    if (group >= memory.serviceUseGroups().size())
      return executorError("explicit memory dispatch has a foreign group");
    const FrozenSpatialMemoryServiceUseGroup &record =
        memory.serviceUseGroups()[group];
    if (record.logicalBinding >= memory.logicalBindings().size())
      return executorError("memory dispatch group has a foreign binding");
    const PnrIndex targetOrdinal =
        candidate.logicalMemoryBinding(record.logicalBinding).target;
    if (targetOrdinal >= memory.bindingTargets().size())
      return executorError("memory dispatch group has a foreign target");
    const FrozenSpatialMemoryBindingTargetOption &target =
        memory.bindingTargets()[targetOrdinal];
    const PnrIndex requiredPattern = explicitMemoryDispatchPatterns_[group];
    const auto members =
        memory.serviceGroupUses().slice(record.useOffset, record.useCount);
    for (PnrIndex member : members) {
      if (member >= explicitMemoryDispatchUseMarks_.size())
        return executorError("memory dispatch group has a foreign member");
      PnrIndex selected = explicitMemoryDispatchUseMarks_[member]
                              ? explicitMemoryDispatchSelections_[member]
                              : candidate.memoryUseDispatch(member);
      if (explicitMemoryDispatchUseMarks_[member] &&
          candidate.memoryUseDispatch(member) != selected) {
        if (llvm::Error error = move.setMemoryUseDispatch(member, selected))
          return error;
      }
      if (patterns[selected] == requiredPattern &&
          detail::memoryDispatchMatchesTarget(
              memory, memory.dispatchOptions()[selected], target))
        continue;
      if (explicitMemoryDispatchUseMarks_[member])
        return intrinsicTransitionFailure(
            "explicit memory-dispatch Action is incompatible with its group");
      auto domain = candidate.memoryDispatchDomain(member);
      if (!domain)
        return domain.takeError();
      selected = getInvalidPnrIndex();
      for (PnrIndex option = (*domain)->optionOffset;
           option < (*domain)->optionOffset + (*domain)->optionCount; ++option)
        if (patterns[option] == requiredPattern &&
            detail::memoryDispatchMatchesTarget(
                memory, memory.dispatchOptions()[option], target)) {
          selected = option;
          break;
        }
      if (selected == getInvalidPnrIndex())
        return intrinsicTransitionFailure(
            "memory-dispatch Action has no group-compatible selection");
      if (llvm::Error error = move.setMemoryUseDispatch(member, selected))
        return error;
    }
  }
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::recordExplicitMemoryExposure(
    const SpatialCandidateState &candidate, PnrIndex exposure,
    PnrIndex option) {
  if (exposure >= explicitMemoryExposureSelections_.size() ||
      option >= candidate.problem().memory().exposureOptions().size())
    return executorError("memory-exposure Action is out of range");
  if (explicitMemoryExposureMarks_[exposure] &&
      explicitMemoryExposureSelections_[exposure] != option)
    return intrinsicTransitionFailure(
        "one ActionBatch selects conflicting memory exposures");
  explicitMemoryExposureMarks_[exposure] = 1;
  explicitMemoryExposureSelections_[exposure] = option;
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::reconcileExplicitMemoryExposures(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  for (PnrIndex exposure = 0; exposure < explicitMemoryExposureMarks_.size();
       ++exposure) {
    if (!explicitMemoryExposureMarks_[exposure])
      continue;
    const PnrIndex option = explicitMemoryExposureSelections_[exposure];
    const PnrIndex binding = memory.exposures()[exposure].logicalBinding;
    const PnrIndex target = candidate.logicalMemoryBinding(binding).target;
    if (target >= memory.bindingTargets().size() ||
        option >= memory.exposureOptions().size())
      return executorError("explicit memory exposure is out of range");
    if (!detail::memoryExposureMatchesTarget(memory.bindingTargets()[target],
                                             memory.exposureOptions()[option]))
      return intrinsicTransitionFailure(
          "explicit memory-exposure Action is incompatible with its binding");
    if (candidate.memoryExposureSelection(exposure) != option)
      if (llvm::Error error = move.setMemoryExposureSelection(exposure, option))
        return error;
  }
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::reconcileBindingRelations(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  if (changedBindingRoots_.empty())
    return llvm::Error::success();
  if (!relationSolver_)
    return executorError("binding relation solver is not prepared");

  const detail::SpatialBindingRelationModel &bindings =
      candidate.problem().bindingRelations();
  relationDecisionQueue_ = changedBindingRoots_;
  for (std::size_t cursor = 0; cursor < relationDecisionQueue_.size();
       ++cursor) {
    const PnrIndex decision = relationDecisionQueue_[cursor];
    for (PnrIndex relation : bindings.decisionRelations(decision))
      for (const detail::InitializerRelationMember &member :
           bindings.relations().members(
               bindings.relations().relations()[relation])) {
        if (member.decision >= relationDecisionMarks_.size())
          return executorError("binding relation member is out of range");
        if (relationDecisionMarks_[member.decision])
          continue;
        relationDecisionMarks_[member.decision] = 1;
        relationDecisionQueue_.push_back(member.decision);
      }
  }

  fixedRelationChoices_ = candidate.bindingRelationChoices_;
  for (PnrIndex decision : relationDecisionQueue_)
    if (decision >= bindings.portDecisionOffset() &&
        !explicitAttachmentMarks_[decision])
      fixedRelationChoices_[decision] = getInvalidPnrIndex();
  auto solved = relationSolver_->solveCanonicalWithFixedChoices(
      candidate.problem()
          .config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed,
      fixedRelationChoices_);
  if (!solved)
    return classifyRelationFailure(solved.takeError());

  for (PnrIndex demand = 0; demand < bindings.portDecisionCount(); ++demand) {
    const PnrIndex selected =
        solved->choices[bindings.portDecisionOffset() + demand];
    const auto choices = bindings.portAttachmentChoices(demand);
    if (selected >= choices.size())
      return executorError("relation solver returned a foreign attachment");
    if (candidate.portAttachment(demand) == choices[selected])
      continue;
    if (llvm::Error error = move.setPortAttachment(demand, choices[selected]))
      return error;
    if (llvm::Error error = markNet(
            candidate.problem().ports().portDemands()[demand].logicalNet))
      return error;
  }
  for (PnrIndex boundary = 0; boundary < bindings.graphBoundaryDecisionCount();
       ++boundary) {
    const PnrIndex selected =
        solved->choices[bindings.graphBoundaryDecisionOffset() + boundary];
    const auto choices = bindings.graphBoundaryAttachmentChoices(boundary);
    if (selected >= choices.size())
      return executorError(
          "relation solver returned a foreign graph-boundary attachment");
    if (candidate.graphBoundaryAttachment(boundary) == choices[selected])
      continue;
    if (llvm::Error error =
            move.setGraphBoundaryAttachment(boundary, choices[selected]))
      return error;
    if (llvm::Error error = markNet(
            candidate.problem().ports().graphBoundaries()[boundary].logicalNet))
      return error;
  }
  return llvm::Error::success();
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
  if (llvm::Error error = router_.beginConstraintSweep(affectedNets_))
    return error;
  for (PnrIndex logicalNet : affectedNets_) {
    if (llvm::Error error = routeCosts_->selectLogicalNet(logicalNet))
      return error;
    llvm::Expected<RouteCost> route = [&]() -> llvm::Expected<RouteCost> {
      switch (pendingRouteKinds_[logicalNet]) {
      case PendingRouteKind::WholeNet:
        return router_.routeWholeNetInMove(move, candidate, *routeCosts_,
                                           logicalNet, endpointLimit);
      case PendingRouteKind::SingleSink:
        return router_.routeSingleSinkInMove(
            move, candidate, *routeCosts_, logicalNet,
            pendingRouteAnchors_[logicalNet], endpointLimit);
      case PendingRouteKind::RootedSubtree:
        return router_.routeRootedSubtreeInMove(
            move, candidate, *routeCosts_, logicalNet,
            pendingRouteAnchors_[logicalNet], endpointLimit);
      }
      llvm_unreachable("unknown pending routing scope");
    }();
    if (!route) {
      if (!admitsUnrouted ||
          pendingRouteKinds_[logicalNet] != PendingRouteKind::WholeNet)
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
    if (llvm::Error error = router_.finishConstraintNet(logicalNet))
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
                                    const SpatialMappingAction &action,
                                    SpatialActionExecutionContext context) {
  return probeBatch(candidate, llvm::ArrayRef<SpatialMappingAction>(&action, 1),
                    context);
}

llvm::Expected<SpatialActionProbe> SpatialActionExecutorScratch::probeBatch(
    SpatialCandidateState &candidate,
    llvm::ArrayRef<SpatialMappingAction> actions,
    SpatialActionExecutionContext context) {
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
  if (llvm::Error error =
          reconcileExplicitLogicalMemoryBindings(move, candidate))
    return restoreAfterFailure(move, std::move(error));
  if (llvm::Error error = reconcileExplicitMemoryExposures(move, candidate))
    return restoreAfterFailure(move, std::move(error));
  if (llvm::Error error = reconcileExplicitMemoryDispatches(move, candidate))
    return restoreAfterFailure(move, std::move(error));
  if (llvm::Error error = reconcileBindingRelations(move, candidate))
    return restoreAfterFailure(move, std::move(error));

  if (globalRouting_ ||
      (context == SpatialActionExecutionContext::FinalClosure &&
       !affectedNets_.empty())) {
    const auto &routing = candidate.problem().config().policy().search.routing;
    llvm::sort(affectedNets_);
    auto closure = router_.routeToClosureInMove(
        move, candidate, *routeCosts_,
        {routing.endpointExpansionLimit, routing.negotiationIterationLimit,
         routing.noProgressIterationLimit, routing.noProgressTrendWindow},
        globalRouting_ ? llvm::ArrayRef<PnrIndex>{}
                       : llvm::ArrayRef<PnrIndex>(affectedNets_),
        {},
        context == SpatialActionExecutionContext::FinalClosure
            ? SpatialRoutingClosureRequirement::Final
            : SpatialRoutingClosureRequirement::PolicyAdmittedTemporary);
    if (!closure)
      return restoreAfterFailure(
          move, classifyTransitionFailure(closure.takeError(), context));
  } else if (llvm::Error error = routeAffectedNets(move, candidate)) {
    return restoreAfterFailure(
        move, classifyTransitionFailure(std::move(error), context));
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
  if (llvm::Error error =
          routeCosts_->synchronizeCandidateTraversals(routeCostTraversals_))
    return restoreAfterFailure(move, std::move(error));

  const bool semanticChange = move.hasSemanticChange();
  dse::ObjectiveVector objective = *currentObjective_;
  dse::ObjectiveSignedDifference difference;
  if (semanticChange) {
    auto evaluated = candidate.problem().objectiveProgram().evaluate(candidate);
    if (!evaluated)
      return restoreAfterFailure(move, evaluated.takeError());
    objective = std::move(*evaluated);
    auto evaluatedDifference =
        candidate.problem().objectiveProgram().selectedEnergyDifference(
            objective, *currentObjective_);
    if (!evaluatedDifference)
      return restoreAfterFailure(move, evaluatedDifference.takeError());
    difference = *evaluatedDifference;
  }

  activeProbe_ = true;
  return SpatialActionProbe(*this, std::move(move), std::move(objective),
                            difference, globalRouting_, semanticChange);
}

std::size_t SpatialActionExecutorScratch::retainedStorageBytes() const {
  return candidateScratch_.retainedStorageBytes() +
         router_.retainedStorageBytes() +
         (routeCosts_ ? routeCosts_->retainedStorageBytes() : 0) +
         retainedBytes(netMarks_) + retainedBytes(affectedNets_) +
         retainedBytes(pendingRouteKinds_) +
         retainedBytes(pendingRouteAnchors_) +
         retainedBytes(routeCostTraversals_) +
         retainedBytes(fixedRelationChoices_) +
         retainedBytes(relationDecisionMarks_) +
         retainedBytes(explicitAttachmentMarks_) +
         retainedBytes(relationDecisionQueue_) +
         retainedBytes(changedBindingRoots_) +
         retainedBytes(explicitLogicalMemorySelections_) +
         retainedBytes(explicitLogicalMemoryMarks_) +
         retainedBytes(explicitLogicalMemoryBindings_) +
         retainedBytes(explicitLogicalMemoryChoices_) +
         retainedBytes(changedLogicalMemoryMarks_) +
         retainedBytes(changedLogicalMemoryBindings_) +
         retainedBytes(explicitMemoryDispatchPatterns_) +
         retainedBytes(explicitMemoryDispatchGroupMarks_) +
         retainedBytes(explicitMemoryDispatchGroups_) +
         retainedBytes(explicitMemoryDispatchSelections_) +
         retainedBytes(explicitMemoryDispatchUseMarks_) +
         retainedBytes(explicitMemoryExposureSelections_) +
         retainedBytes(explicitMemoryExposureMarks_) +
         (memoryConstraintScratch_
              ? memoryConstraintScratch_->retainedStorageBytes()
              : 0) +
         (relationSolver_ ? relationSolver_->retainedStorageBytes() : 0);
}
