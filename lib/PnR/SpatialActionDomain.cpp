#include "PnR/SpatialActionDomain.h"

#include "SpatialBindingRelationModel.h"

#include "PnR/PnrIndex.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <system_error>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_action_domain_invalid: %s", detail.str().c_str());
}

llvm::Expected<PnrIndex> actionIndex(std::size_t value, llvm::StringRef table,
                                     PnrCapacityMeasure measure) {
  return checkedPnrIndex({"SpatialActionDomain", table, "Action", measure},
                         value);
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

llvm::Error
SpatialActionDomainScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  const detail::SpatialBindingRelationModel &relations =
      problem.bindingRelations();
  if (relations.deferredProjection())
    return invalid("binding relation owner is incomplete");

  const auto decisionOffsets = relations.relations().decisionChoiceOffsets();
  const std::size_t realizationChoiceCapacity =
      decisionOffsets.empty()
          ? 0
          : decisionOffsets[relations.realizationDecisionCount()];
  const std::size_t realizationAnchorCapacity =
      relations.realizationDecisionCount();
  const std::size_t logicalNetCount = problem.transfers().logicalNets().size();
  const std::size_t sinkCount = problem.transfers().logicalNetSinks().size();
  const std::size_t endpointCount = problem.routing().routingEndpoints().size();
  const PnrCapacityContext transportCapacity{"SpatialActionDomain",
                                             "transportChoices", "Action",
                                             PnrCapacityMeasure::Count};
  auto netEndpointCapacity = checkedPnrIndexMultiply(
      transportCapacity, logicalNetCount, endpointCount);
  if (!netEndpointCapacity)
    return netEndpointCapacity.takeError();
  auto transportChoiceCapacity =
      checkedPnrIndexAdd(transportCapacity, logicalNetCount, sinkCount);
  if (!transportChoiceCapacity)
    return transportChoiceCapacity.takeError();
  transportChoiceCapacity = checkedPnrIndexAdd(
      transportCapacity, *transportChoiceCapacity, *netEndpointCapacity);
  if (!transportChoiceCapacity)
    return transportChoiceCapacity.takeError();
  transportChoiceCapacity = checkedPnrIndexAdd(
      transportCapacity, *transportChoiceCapacity, sinkCount);
  if (!transportChoiceCapacity)
    return transportChoiceCapacity.takeError();
  transportChoiceCapacity =
      checkedPnrIndexAdd(transportCapacity, *transportChoiceCapacity,
                         problem.resources().capacityDimensions().size());
  if (!transportChoiceCapacity)
    return transportChoiceCapacity.takeError();
  transportChoiceCapacity = checkedPnrIndexAdd(
      transportCapacity, *transportChoiceCapacity, *netEndpointCapacity);
  if (!transportChoiceCapacity)
    return transportChoiceCapacity.takeError();
  transportChoiceCapacity = checkedPnrIndexAdd(
      transportCapacity, *transportChoiceCapacity,
      problem.routing().tagContinuity().matchDomains().size());
  if (!transportChoiceCapacity)
    return transportChoiceCapacity.takeError();
  transportChoiceCapacity =
      checkedPnrIndexAdd(transportCapacity, *transportChoiceCapacity,
                         logicalNetCount == 0 ? 0 : 1);
  if (!transportChoiceCapacity)
    return transportChoiceCapacity.takeError();
  auto transportAnchorCapacity =
      checkedPnrIndexAdd(transportCapacity, logicalNetCount, sinkCount);
  if (!transportAnchorCapacity)
    return transportAnchorCapacity.takeError();
  transportAnchorCapacity =
      checkedPnrIndexAdd(transportCapacity, *transportAnchorCapacity,
                         problem.resources().capacityDimensions().size());
  if (!transportAnchorCapacity)
    return transportAnchorCapacity.takeError();
  transportAnchorCapacity = checkedPnrIndexAdd(
      transportCapacity, *transportAnchorCapacity, *netEndpointCapacity);
  if (!transportAnchorCapacity)
    return transportAnchorCapacity.takeError();
  transportAnchorCapacity = checkedPnrIndexAdd(
      transportCapacity, *transportAnchorCapacity,
      problem.routing().tagContinuity().matchDomains().size());
  if (!transportAnchorCapacity)
    return transportAnchorCapacity.takeError();
  transportAnchorCapacity =
      checkedPnrIndexAdd(transportCapacity, *transportAnchorCapacity,
                         logicalNetCount == 0 ? 0 : 1);
  if (!transportAnchorCapacity)
    return transportAnchorCapacity.takeError();
  auto resourceChoiceCapacity =
      checkedPnrIndexAdd({"SpatialActionDomain", "resourceChoices", "Action",
                          PnrCapacityMeasure::Count},
                         problem.ports().attachmentOptions().size(),
                         problem.handshake().memoryOperationPlans().size());
  if (!resourceChoiceCapacity)
    return resourceChoiceCapacity.takeError();
  auto resourceAnchorCapacity =
      checkedPnrIndexAdd({"SpatialActionDomain", "resourceAnchors", "Action",
                          PnrCapacityMeasure::Count},
                         problem.ports().portDemands().size(),
                         problem.ports().graphBoundaries().size());
  if (!resourceAnchorCapacity)
    return resourceAnchorCapacity.takeError();
  resourceAnchorCapacity = checkedPnrIndexAdd(
      {"SpatialActionDomain", "resourceAnchors", "Action",
       PnrCapacityMeasure::Count},
      *resourceAnchorCapacity, problem.realizations().memoryActors().size());
  if (!resourceAnchorCapacity)
    return resourceAnchorCapacity.takeError();

  realizationAnchors_.clear();
  realizationChoices_.clear();
  transportAnchors_.clear();
  transportChoices_.clear();
  routeRootEndpoints_.clear();
  resourceAnchors_.clear();
  resourceChoices_.clear();
  movableDecisionCount_ = 0;
  realizationAnchors_.reserve(realizationAnchorCapacity);
  realizationChoices_.reserve(realizationChoiceCapacity);
  transportAnchors_.reserve(*transportAnchorCapacity);
  transportChoices_.reserve(*transportChoiceCapacity);
  routeRootEndpoints_.reserve(endpointCount);
  resourceAnchors_.reserve(*resourceAnchorCapacity);
  resourceChoices_.reserve(*resourceChoiceCapacity);
  relationChoices_.resize(relations.decisionCount());
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

llvm::Error
SpatialActionDomainScratch::rebuild(const SpatialCandidateState &candidate) {
  if (preparedProblem_ == nullptr)
    return invalid("scratch storage was not prepared");
  if (&candidate.problem() != preparedProblem_)
    return invalid("candidate belongs to a different Frozen problem");

  realizationAnchors_.clear();
  realizationChoices_.clear();
  transportAnchors_.clear();
  transportChoices_.clear();
  routeRootEndpoints_.clear();
  resourceAnchors_.clear();
  resourceChoices_.clear();
  movableDecisionCount_ = 0;

  const auto currentRelationChoices =
      llvm::ArrayRef(candidate.bindingRelationChoices_);
  if (currentRelationChoices.size() != relationChoices_.size())
    return invalid("candidate binding relation projection is malformed");
  std::copy(currentRelationChoices.begin(), currentRelationChoices.end(),
            relationChoices_.begin());

  const detail::SpatialBindingRelationModel &relations =
      preparedProblem_->bindingRelations();
  const auto appendRealizationRange = [&](std::size_t offset) -> llvm::Error {
    if (realizationChoices_.size() == offset)
      return llvm::Error::success();
    auto checkedOffset =
        actionIndex(offset, "realizationChoices", PnrCapacityMeasure::Offset);
    if (!checkedOffset)
      return checkedOffset.takeError();
    auto checkedCount =
        actionIndex(realizationChoices_.size() - offset, "realizationChoices",
                    PnrCapacityMeasure::Count);
    if (!checkedCount)
      return checkedCount.takeError();
    realizationAnchors_.push_back({*checkedOffset, *checkedCount});
    if (movableDecisionCount_ == std::numeric_limits<std::uint64_t>::max())
      return invalid("movable decision count overflows u64");
    ++movableDecisionCount_;
    return llvm::Error::success();
  };
  const auto relationChoiceIsLegal =
      [&](PnrIndex decision, PnrIndex localChoice, bool constraintsOnly) {
        const PnrIndex oldChoice = relationChoices_[decision];
        relationChoices_[decision] = localChoice;
        const bool legal = llvm::all_of(
            relations.decisionRelations(decision), [&](PnrIndex relation) {
              if (constraintsOnly && !relations.relationIsConstraint(relation))
                return true;
              return relations.relationSatisfied(relation, relationChoices_);
            });
        relationChoices_[decision] = oldChoice;
        return legal;
      };

  const auto &realizations = preparedProblem_->realizations();
  for (PnrIndex realization = 0;
       realization < realizations.computeRealizations().size(); ++realization) {
    const std::size_t offset = realizationChoices_.size();
    const auto choices = relations.computeChoices(realization);
    for (auto [localChoice, choice] : llvm::enumerate(choices)) {
      const auto &current = candidate.computeBinding(realization);
      if ((current.placement == choice.placement &&
           current.instructionContext == choice.instructionContext) ||
          !relationChoiceIsLegal(realization,
                                 static_cast<PnrIndex>(localChoice), true))
        continue;
      realizationChoices_.emplace_back(SpatialComputeBindingAction{
          realization, choice.placement, choice.instructionContext});
    }
    if (llvm::Error error = appendRealizationRange(offset))
      return error;
  }
  const PnrIndex memoryDecisionOffset = relations.computeDecisionCount();
  for (PnrIndex realization = 0;
       realization < realizations.memoryRealizations().size(); ++realization) {
    const std::size_t offset = realizationChoices_.size();
    const auto choices = relations.memoryChoices(realization);
    for (auto [localChoice, choice] : llvm::enumerate(choices)) {
      if (candidate.memoryBinding(realization).placement == choice.placement ||
          !relationChoiceIsLegal(memoryDecisionOffset + realization,
                                 static_cast<PnrIndex>(localChoice), true))
        continue;
      realizationChoices_.emplace_back(
          SpatialMemoryBindingAction{realization, choice.placement});
    }
    if (llvm::Error error = appendRealizationRange(offset))
      return error;
  }

  const auto appendTransportRange = [&](std::size_t offset) -> llvm::Error {
    if (transportChoices_.size() == offset)
      return llvm::Error::success();
    auto checkedOffset =
        actionIndex(offset, "transportChoices", PnrCapacityMeasure::Offset);
    if (!checkedOffset)
      return checkedOffset.takeError();
    auto checkedCount =
        actionIndex(transportChoices_.size() - offset, "transportChoices",
                    PnrCapacityMeasure::Count);
    if (!checkedCount)
      return checkedCount.takeError();
    transportAnchors_.push_back({*checkedOffset, *checkedCount});
    return llvm::Error::success();
  };
  const auto appendWitness = [&](ResolvedPnrViolationKind kind,
                                 PnrIndex ordinal) -> llvm::Error {
    const std::size_t offset = transportChoices_.size();
    transportChoices_.emplace_back(
        SpatialWitnessRegionRoutingAction{kind, ordinal});
    return appendTransportRange(offset);
  };
  const auto &transfers = preparedProblem_->transfers();
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet) {
    const std::size_t offset = transportChoices_.size();
    transportChoices_.emplace_back(SpatialWholeNetRoutingAction{logicalNet});
    const RouteTreeState &route = candidate.routeTree(logicalNet);
    if (route.isRouted()) {
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
      for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
        transportChoices_.emplace_back(
            SpatialSingleSinkRoutingAction{logicalNet, sink});
      const PnrIndex source = *route.sourceEndpoint();
      routeRootEndpoints_.clear();
      for (const RouteTreeNode &node : route.nodeStorage())
        if (node.isActive() && node.endpoint != source)
          routeRootEndpoints_.push_back(node.endpoint);
      llvm::sort(routeRootEndpoints_);
      for (PnrIndex endpoint : routeRootEndpoints_)
        transportChoices_.emplace_back(
            SpatialRootedSubtreeRoutingAction{logicalNet, endpoint});
    }
    if (llvm::Error error = appendTransportRange(offset))
      return error;
  }
  if (preparedProblem_->transfers().logicalNets().size() >
      std::numeric_limits<std::uint64_t>::max() - movableDecisionCount_)
    return invalid("movable decision count overflows u64");
  movableDecisionCount_ += preparedProblem_->transfers().logicalNets().size();

  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet) {
    if (candidate.routeTree(logicalNet).isRouted())
      continue;
    const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
    for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
      if (llvm::Error error =
              appendWitness(ResolvedPnrViolationKind::UnroutedObligation,
                            net.sinkOffset + sink))
        return error;
  }
  for (PnrIndex capacity = 0;
       capacity < preparedProblem_->resources().capacityDimensions().size();
       ++capacity)
    if (candidate.routeCapacityOveruseRaw(capacity) != 0)
      if (llvm::Error error = appendWitness(
              ResolvedPnrViolationKind::CapacityOveruse, capacity))
        return error;

  std::size_t globalSegment = 0;
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet) {
    const auto values = candidate.tagValues(logicalNet);
    for (const auto &value : values) {
      auto ordinal =
          actionIndex(globalSegment, "tagSegments", PnrCapacityMeasure::Offset);
      if (!ordinal)
        return ordinal.takeError();
      if (!value)
        if (llvm::Error error = appendWitness(
                ResolvedPnrViolationKind::TagUnassigned, *ordinal))
          return error;
      ++globalSegment;
    }
  }
  for (PnrIndex domain = 0;
       domain <
       preparedProblem_->routing().tagContinuity().matchDomains().size();
       ++domain)
    if (candidate.tagDomainConflictCount(domain) != 0)
      if (llvm::Error error =
              appendWitness(ResolvedPnrViolationKind::TagConflict, domain))
        return error;

  if (!transfers.logicalNets().empty()) {
    const std::size_t offset = transportChoices_.size();
    transportChoices_.emplace_back(SpatialGlobalRoutingAction{});
    if (llvm::Error error = appendTransportRange(offset))
      return error;
  }

  const auto appendResourceRange = [&](std::size_t offset) -> llvm::Error {
    if (resourceChoices_.size() == offset)
      return llvm::Error::success();
    auto checkedOffset =
        actionIndex(offset, "resourceChoices", PnrCapacityMeasure::Offset);
    if (!checkedOffset)
      return checkedOffset.takeError();
    auto checkedCount =
        actionIndex(resourceChoices_.size() - offset, "resourceChoices",
                    PnrCapacityMeasure::Count);
    if (!checkedCount)
      return checkedCount.takeError();
    resourceAnchors_.push_back({*checkedOffset, *checkedCount});
    if (movableDecisionCount_ == std::numeric_limits<std::uint64_t>::max())
      return invalid("movable decision count overflows u64");
    ++movableDecisionCount_;
    return llvm::Error::success();
  };

  const auto &ports = preparedProblem_->ports();
  for (PnrIndex demand = 0; demand < ports.portDemands().size(); ++demand) {
    const std::size_t offset = resourceChoices_.size();
    const FrozenSpatialPortDemand &record = ports.portDemands()[demand];
    const bool compute = record.kind == FrozenSpatialPortDemandKind::Compute;
    const PnrIndex placement =
        compute ? candidate.computeBinding(record.realization).placement
                : candidate.memoryBinding(record.realization).placement;
    const PnrIndex ownerOffset =
        compute ? realizations.computeRealizations()[record.realization]
                      .placementOffset
                : realizations.memoryRealizations()[record.realization]
                      .placementOffset;
    if (placement < ownerOffset ||
        placement - ownerOffset >= record.placementDomainCount)
      return invalid("candidate port placement has no exact local domain");
    const FrozenSpatialPortPlacementDomain &domain =
        ports.placementDomains()[record.placementDomainOffset + placement -
                                 ownerOffset];
    for (PnrIndex local = 0; local < domain.attachmentOptionCount; ++local) {
      const PnrIndex option = domain.attachmentOptionOffset + local;
      const auto localChoice =
          relations.portAttachmentChoiceOrdinal(demand, option);
      if (candidate.portAttachment(demand) != option && localChoice &&
          relationChoiceIsLegal(relations.portDecisionOffset() + demand,
                                *localChoice, false))
        resourceChoices_.emplace_back(
            SpatialPortAttachmentAction{demand, option});
    }
    if (llvm::Error error = appendResourceRange(offset))
      return error;
  }
  for (PnrIndex boundary = 0; boundary < ports.graphBoundaries().size();
       ++boundary) {
    const std::size_t offset = resourceChoices_.size();
    const FrozenSpatialGraphBoundary &record =
        ports.graphBoundaries()[boundary];
    for (PnrIndex local = 0; local < record.attachmentOptionCount; ++local) {
      const PnrIndex option = record.attachmentOptionOffset + local;
      const auto localChoice =
          relations.graphBoundaryAttachmentChoiceOrdinal(boundary, option);
      if (candidate.graphBoundaryAttachment(boundary) != option &&
          localChoice &&
          relationChoiceIsLegal(relations.graphBoundaryDecisionOffset() +
                                    boundary,
                                *localChoice, false))
        resourceChoices_.emplace_back(
            SpatialGraphBoundaryAttachmentAction{boundary, option});
    }
    if (llvm::Error error = appendResourceRange(offset))
      return error;
  }

  const auto &handshake = preparedProblem_->handshake();
  for (PnrIndex actor = 0; actor < realizations.memoryActors().size();
       ++actor) {
    const std::size_t offset = resourceChoices_.size();
    const PnrIndex realization = realizations.memoryActorRealizations()[actor];
    const FrozenSpatialMemoryRealization &owner =
        realizations.memoryRealizations()[realization];
    const PnrIndex placement = candidate.memoryBinding(realization).placement;
    const PnrIndex domainOffset =
        handshake.memoryPlacementDomainOffsets()[placement];
    const FrozenSpatialMemoryOperationHandshakeDomain &domain =
        handshake
            .memoryOperationDomains()[domainOffset + actor - owner.actorOffset];
    for (PnrIndex local = 0; local < domain.planCount; ++local) {
      const PnrIndex plan = domain.planOffset + local;
      if (candidate.memoryOperationPlan(actor) != plan)
        resourceChoices_.emplace_back(
            SpatialMemoryOperationPlanAction{actor, plan});
    }
    if (llvm::Error error = appendResourceRange(offset))
      return error;
  }

  return llvm::Error::success();
}

SpatialActionProposalDomain SpatialActionDomainScratch::view() const {
  return {realizationAnchors_, realizationChoices_, transportAnchors_,
          transportChoices_,   resourceAnchors_,    resourceChoices_};
}

std::size_t SpatialActionDomainScratch::retainedStorageBytes() const {
  return retainedBytes(realizationAnchors_) +
         retainedBytes(realizationChoices_) + retainedBytes(transportAnchors_) +
         retainedBytes(transportChoices_) + retainedBytes(resourceAnchors_) +
         retainedBytes(resourceChoices_) + retainedBytes(relationChoices_) +
         retainedBytes(routeRootEndpoints_);
}
