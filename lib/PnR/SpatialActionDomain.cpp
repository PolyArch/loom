#include "PnR/SpatialActionDomain.h"

#include "SpatialActionProposalInternal.h"
#include "SpatialProgressIndex.h"

#include "SpatialBindingRelationModel.h"
#include "SpatialMemoryCompatibility.h"
#include "SpatialMemoryConstraintModel.h"

#include "PnR/PnrIndex.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
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

SpatialActionDomainScratch::SpatialActionDomainScratch() = default;
SpatialActionDomainScratch::~SpatialActionDomainScratch() = default;

llvm::Error
SpatialActionDomainScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  const detail::SpatialBindingRelationModel &relations =
      problem.bindingRelations();
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
  auto initialResourceChoiceCapacity =
      checkedPnrIndexAdd({"SpatialActionDomain", "resourceChoices", "Action",
                          PnrCapacityMeasure::Count},
                         problem.ports().attachmentOptions().size(),
                         problem.handshake().memoryOperationPlans().size());
  if (!initialResourceChoiceCapacity)
    return initialResourceChoiceCapacity.takeError();
  PnrIndex resourceChoiceCapacity = *initialResourceChoiceCapacity;
  const PnrCapacityContext resourceChoiceContext{"SpatialActionDomain",
                                                 "resourceChoices", "Action",
                                                 PnrCapacityMeasure::Count};
  const auto growResourceChoices = [&](std::size_t count) -> llvm::Error {
    auto grown = checkedPnrIndexAdd(resourceChoiceContext,
                                    resourceChoiceCapacity, count);
    if (!grown)
      return grown.takeError();
    resourceChoiceCapacity = *grown;
    return llvm::Error::success();
  };

  PnrIndex maximumLogicalMemoryChoiceCapacity = 0;
  for (PnrIndex binding = 0;
       binding < problem.memory().logicalBindings().size(); ++binding) {
    auto capacity =
        problem.memoryConstraints().logicalBindingChoiceCapacity(binding);
    if (!capacity)
      return capacity.takeError();
    maximumLogicalMemoryChoiceCapacity =
        std::max(maximumLogicalMemoryChoiceCapacity, *capacity);
    if (llvm::Error error = growResourceChoices(*capacity))
      return error;
  }

  const auto &realizations = problem.realizations();
  const auto &memory = problem.memory();
  for (const FrozenSpatialMemoryRootedUse &use : memory.rootedUses()) {
    if (use.actor >= realizations.memoryActorRealizations().size())
      return invalid("rooted memory use has a foreign actor");
    const PnrIndex realization =
        realizations.memoryActorRealizations()[use.actor];
    if (realization >= realizations.memoryRealizations().size())
      return invalid("rooted memory use has a foreign realization");
    const FrozenSpatialMemoryRealization &owner =
        realizations.memoryRealizations()[realization];
    if (use.actor < owner.actorOffset ||
        use.actor - owner.actorOffset >= owner.actorCount)
      return invalid("rooted memory use is outside its actor slice");
    const PnrIndex localActor = use.actor - owner.actorOffset;
    PnrIndex maximumOptions = 0;
    for (PnrIndex placement = owner.placementOffset;
         placement < owner.placementOffset + owner.placementCount;
         ++placement) {
      const PnrIndex domainOffset =
          memory.memoryPlacementDomainOffsets()[placement];
      const FrozenSpatialMemoryDispatchDomain &domain =
          memory.dispatchDomains()[domainOffset + localActor];
      maximumOptions = std::max(maximumOptions, domain.optionCount);
    }
    if (llvm::Error error = growResourceChoices(maximumOptions))
      return error;
  }
  auto exposureChoiceCapacity =
      checkedPnrIndexMultiply(resourceChoiceContext, memory.exposures().size(),
                              memory.exposureOptions().size());
  if (!exposureChoiceCapacity)
    return exposureChoiceCapacity.takeError();
  if (llvm::Error error = growResourceChoices(*exposureChoiceCapacity))
    return error;

  auto initialResourceAnchorCapacity =
      checkedPnrIndexAdd({"SpatialActionDomain", "resourceAnchors", "Action",
                          PnrCapacityMeasure::Count},
                         problem.ports().portDemands().size(),
                         problem.ports().graphBoundaries().size());
  if (!initialResourceAnchorCapacity)
    return initialResourceAnchorCapacity.takeError();
  PnrIndex resourceAnchorCapacity = *initialResourceAnchorCapacity;
  const PnrCapacityContext resourceAnchorContext{"SpatialActionDomain",
                                                 "resourceAnchors", "Action",
                                                 PnrCapacityMeasure::Count};
  for (std::size_t count :
       {problem.realizations().memoryActors().size(),
        memory.logicalBindings().size(), memory.rootedUses().size(),
        memory.exposures().size()}) {
    auto grown = checkedPnrIndexAdd(resourceAnchorContext,
                                    resourceAnchorCapacity, count);
    if (!grown)
      return grown.takeError();
    resourceAnchorCapacity = *grown;
  }

  realizationAnchors_.clear();
  realizationChoices_.clear();
  transportAnchors_.clear();
  transportChoices_.clear();
  routeRootEndpoints_.clear();
  routeSubtreeSlots_.clear();
  routeSubtreeHasSink_.clear();
  hardProgressWitnessOwners_.clear();
  resourceAnchors_.clear();
  resourceChoices_.clear();
  realizationMovableDecisionCount_ = 0;
  transportMovableDecisionCount_ = 0;
  resourceMovableDecisionCount_ = 0;
  realizationAnchors_.reserve(realizationAnchorCapacity);
  realizationChoices_.reserve(realizationChoiceCapacity);
  transportAnchors_.reserve(*transportAnchorCapacity);
  transportChoices_.reserve(*transportChoiceCapacity);
  routeRootEndpoints_.reserve(endpointCount);
  routeSubtreeSlots_.reserve(endpointCount);
  routeSubtreeHasSink_.reserve(endpointCount);
  hardProgressWitnessOwners_.reserve(
      problem.progressIndex().finiteBufferOwners().size());
  resourceAnchors_.reserve(resourceAnchorCapacity);
  resourceChoices_.reserve(resourceChoiceCapacity);
  relationChoices_.resize(relations.decisionCount());
  relationValueOffsets_.clear();
  relationValues_.clear();
  relationDistinctValueCounts_.clear();
  rootClosedRelations_.clear();
  relationValueOffsets_.reserve(relations.relations().relations().size() + 1);
  relationDistinctValueCounts_.reserve(
      relations.relations().relations().size());
  rootClosedRelations_.reserve(relations.relations().relations().size());
  relationValueOffsets_.push_back(0);
  const auto decisionChoiceOffsets =
      relations.relations().decisionChoiceOffsets();
  for (const detail::InitializerRelationRecord &relation :
       relations.relations().relations()) {
    std::vector<PnrIndex> values;
    bool rootClosed = true;
    for (const detail::InitializerRelationMember &member :
         relations.relations().members(relation)) {
      rootClosed &= member.decision < relations.portDecisionOffset();
      const PnrIndex choiceCount = decisionChoiceOffsets[member.decision + 1] -
                                   decisionChoiceOffsets[member.decision];
      values.reserve(values.size() + choiceCount);
      for (PnrIndex choice = 0; choice < choiceCount; ++choice)
        values.push_back(relations.relations().projectedValue(member, choice));
    }
    llvm::sort(values);
    values.erase(std::unique(values.begin(), values.end()), values.end());
    if (values.empty())
      return invalid("binding relation has no projected values");
    if (values.size() > relationValues_.max_size() - relationValues_.size())
      return invalid("binding relation value index exceeds host size_t");
    relationValues_.insert(relationValues_.end(), values.begin(), values.end());
    relationValueOffsets_.push_back(relationValues_.size());
    relationDistinctValueCounts_.push_back(0);
    rootClosedRelations_.push_back(rootClosed ? 1 : 0);
  }
  relationValueLoads_.resize(relationValues_.size());
  relationDecisionMemberOffsets_.clear();
  relationDecisionMembers_.clear();
  relationDecisionMemberOffsets_.reserve(
      relations.relations().decisionRelations().size() + 1);
  relationDecisionMemberOffsets_.push_back(0);
  const auto relationIncidenceOffsets =
      relations.relations().decisionRelationOffsets();
  const auto relationIncidences = relations.relations().decisionRelations();
  for (PnrIndex decision = 0; decision < relations.decisionCount();
       ++decision) {
    for (PnrIndex incidence = relationIncidenceOffsets[decision];
         incidence < relationIncidenceOffsets[decision + 1]; ++incidence) {
      const PnrIndex relationOrdinal = relationIncidences[incidence];
      for (const detail::InitializerRelationMember &member :
           relations.relations().members(
               relations.relations().relations()[relationOrdinal]))
        if (member.decision == decision)
          relationDecisionMembers_.push_back(
              {member.projectedValueOffset, member.demand});
      if (relationDecisionMemberOffsets_.back() ==
          relationDecisionMembers_.size())
        return invalid("binding relation incidence has no matching member");
      relationDecisionMemberOffsets_.push_back(relationDecisionMembers_.size());
    }
  }
  if (relationDecisionMemberOffsets_.size() != relationIncidences.size() + 1)
    return invalid("binding relation incidence index is incomplete");
  logicalMemoryChoices_.resize(maximumLogicalMemoryChoiceCapacity);
  if (!memoryConstraintScratch_)
    memoryConstraintScratch_ =
        std::make_unique<detail::SpatialMemoryConstraintScratch>();
  if (llvm::Error error =
          problem.memoryConstraints().prepareScratch(*memoryConstraintScratch_))
    return error;
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
  hardProgressWitnessOwners_.clear();
  resourceAnchors_.clear();
  resourceChoices_.clear();
  realizationMovableDecisionCount_ = 0;
  transportMovableDecisionCount_ = 0;
  resourceMovableDecisionCount_ = 0;
  examinedRealizationChoiceCount_ = 0;
  fixedRelationPrunedRealizationChoiceCount_ = 0;

  const auto currentRelationChoices =
      llvm::ArrayRef(candidate.bindingRelationChoices_);
  if (currentRelationChoices.size() != relationChoices_.size())
    return invalid("candidate binding relation projection is malformed");
  std::copy(currentRelationChoices.begin(), currentRelationChoices.end(),
            relationChoices_.begin());

  const detail::SpatialBindingRelationModel &relations =
      preparedProblem_->bindingRelations();
  const detail::InitializerRelationModel &relationModel = relations.relations();
  if (relationValueOffsets_.size() != relationModel.relations().size() + 1 ||
      relationDistinctValueCounts_.size() != relationModel.relations().size() ||
      rootClosedRelations_.size() != relationModel.relations().size() ||
      relationValueLoads_.size() != relationValues_.size() ||
      relationDecisionMemberOffsets_.size() !=
          relationModel.decisionRelations().size() + 1)
    return invalid("binding relation scratch shape is malformed");
  std::fill(relationValueLoads_.begin(), relationValueLoads_.end(), 0);
  std::fill(relationDistinctValueCounts_.begin(),
            relationDistinctValueCounts_.end(), 0);
  for (PnrIndex relationOrdinal = 0;
       relationOrdinal < relationModel.relations().size(); ++relationOrdinal) {
    const detail::InitializerRelationRecord &relation =
        relationModel.relations()[relationOrdinal];
    const auto values = llvm::ArrayRef(relationValues_)
                            .slice(relationValueOffsets_[relationOrdinal],
                                   relationValueOffsets_[relationOrdinal + 1] -
                                       relationValueOffsets_[relationOrdinal]);
    auto loads = llvm::MutableArrayRef(relationValueLoads_)
                     .slice(relationValueOffsets_[relationOrdinal],
                            relationValueOffsets_[relationOrdinal + 1] -
                                relationValueOffsets_[relationOrdinal]);
    const auto capacities = relationModel.valueCapacities(relation);
    for (const detail::InitializerRelationMember &member :
         relationModel.members(relation)) {
      const PnrIndex rawValue = relationModel.projectedValue(
          member, relationChoices_[member.decision]);
      const auto found = llvm::lower_bound(values, rawValue);
      if (found == values.end() || *found != rawValue)
        return invalid("binding relation value index is incomplete");
      const std::size_t value =
          static_cast<std::size_t>(found - values.begin());
      const std::uint64_t demand =
          relation.kind == detail::InitializerRelationKind::Capacity
              ? member.demand
              : 1;
      if (loads[value] == 0)
        ++relationDistinctValueCounts_[relationOrdinal];
      if (demand > std::numeric_limits<std::uint64_t>::max() - loads[value])
        return invalid("binding relation load overflows u64");
      loads[value] += demand;
      if (relation.kind == detail::InitializerRelationKind::Disjoint &&
          loads[value] != 1)
        return invalid("candidate violates a disjoint binding relation");
      if (relation.kind == detail::InitializerRelationKind::Capacity &&
          (value >= capacities.size() || loads[value] > capacities[value]))
        return invalid("candidate violates a binding capacity relation");
    }
    if (relation.kind == detail::InitializerRelationKind::Equal &&
        relationDistinctValueCounts_[relationOrdinal] != 1)
      return invalid("candidate violates an equal binding relation");
  }
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
    if (realizationMovableDecisionCount_ ==
        std::numeric_limits<std::uint64_t>::max())
      return invalid("movable decision count overflows u64");
    ++realizationMovableDecisionCount_;
    return llvm::Error::success();
  };
  const auto relationChoiceIsLegal = [&](PnrIndex decision,
                                         PnrIndex localChoice,
                                         bool constraintsOnly) {
    const PnrIndex oldChoice = relationChoices_[decision];
    const auto decisionRelationOffsets =
        relationModel.decisionRelationOffsets();
    for (auto incidenceRecord :
         llvm::enumerate(relations.decisionRelations(decision))) {
      const std::size_t localIncidence = incidenceRecord.index();
      const PnrIndex relationOrdinal = incidenceRecord.value();
      if (constraintsOnly && !relations.relationIsConstraint(relationOrdinal) &&
          !rootClosedRelations_[relationOrdinal])
        continue;
      const detail::InitializerRelationRecord &relation =
          relationModel.relations()[relationOrdinal];
      const auto values =
          llvm::ArrayRef(relationValues_)
              .slice(relationValueOffsets_[relationOrdinal],
                     relationValueOffsets_[relationOrdinal + 1] -
                         relationValueOffsets_[relationOrdinal]);
      auto loads = llvm::MutableArrayRef(relationValueLoads_)
                       .slice(relationValueOffsets_[relationOrdinal],
                              relationValueOffsets_[relationOrdinal + 1] -
                                  relationValueOffsets_[relationOrdinal]);
      const auto capacities = relationModel.valueCapacities(relation);
      const std::size_t incidence =
          decisionRelationOffsets[decision] + localIncidence;
      const auto changedMembers =
          llvm::ArrayRef(relationDecisionMembers_)
              .slice(relationDecisionMemberOffsets_[incidence],
                     relationDecisionMemberOffsets_[incidence + 1] -
                         relationDecisionMemberOffsets_[incidence]);
      const auto update = [&](PnrIndex choice, bool add) {
        bool legal = true;
        for (const RelationDecisionMember &record : changedMembers) {
          const detail::InitializerRelationMember member{
              decision, record.projectedValueOffset, record.demand};
          const PnrIndex rawValue =
              relationModel.projectedValue(member, choice);
          const auto found = llvm::lower_bound(values, rawValue);
          assert(found != values.end() && *found == rawValue);
          const std::size_t value =
              static_cast<std::size_t>(found - values.begin());
          const std::uint64_t demand =
              relation.kind == detail::InitializerRelationKind::Capacity
                  ? record.demand
                  : 1;
          if (add) {
            assert(demand <=
                   std::numeric_limits<std::uint64_t>::max() - loads[value]);
            if (loads[value] == 0)
              ++relationDistinctValueCounts_[relationOrdinal];
            loads[value] += demand;
            if (relation.kind == detail::InitializerRelationKind::Disjoint &&
                loads[value] != 1)
              legal = false;
            if (relation.kind == detail::InitializerRelationKind::Capacity &&
                loads[value] > capacities[value])
              legal = false;
          } else {
            assert(loads[value] >= demand);
            loads[value] -= demand;
            if (loads[value] == 0)
              --relationDistinctValueCounts_[relationOrdinal];
          }
        }
        return legal;
      };
      update(oldChoice, false);
      bool legal = update(localChoice, true);
      if (relation.kind == detail::InitializerRelationKind::Equal &&
          relationDistinctValueCounts_[relationOrdinal] != 1)
        legal = false;
      update(localChoice, false);
      update(oldChoice, true);
      if (!legal)
        return false;
    }
    return true;
  };

  const auto &realizations = preparedProblem_->realizations();
  for (PnrIndex realization = 0;
       realization < realizations.computeRealizations().size(); ++realization) {
    const std::size_t offset = realizationChoices_.size();
    const auto choices = relations.computeChoices(realization);
    for (auto [localChoice, choice] : llvm::enumerate(choices)) {
      const auto &current = candidate.computeBinding(realization);
      if (current.placement == choice.placement &&
          current.instructionContext == choice.instructionContext)
        continue;
      ++examinedRealizationChoiceCount_;
      if (!relationChoiceIsLegal(realization,
                                 static_cast<PnrIndex>(localChoice), true)) {
        ++fixedRelationPrunedRealizationChoiceCount_;
        continue;
      }
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
      if (candidate.memoryBinding(realization).placement == choice.placement)
        continue;
      ++examinedRealizationChoiceCount_;
      if (!relationChoiceIsLegal(memoryDecisionOffset + realization,
                                 static_cast<PnrIndex>(localChoice), true)) {
        ++fixedRelationPrunedRealizationChoiceCount_;
        continue;
      }
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
  std::uint64_t externalNetCount = 0;
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet) {
    if (candidate.usesRegisterFifo(logicalNet))
      continue;
    ++externalNetCount;
    const std::size_t offset = transportChoices_.size();
    transportChoices_.emplace_back(SpatialWholeNetRoutingAction{logicalNet});
    const RouteTreeState &route = candidate.routeTree(logicalNet);
    if (route.isRouted()) {
      const FrozenSpatialLogicalNet &net = transfers.logicalNets()[logicalNet];
      for (PnrIndex sink = 0; sink < net.sinkCount; ++sink)
        transportChoices_.emplace_back(
            SpatialSingleSinkRoutingAction{logicalNet, sink});
      const PnrIndex source = *route.sourceEndpoint();
      const auto sourceSlot = route.findNode(source);
      if (!sourceSlot)
        return invalid("routed tree has no source node");
      routeRootEndpoints_.clear();
      routeSubtreeSlots_.clear();
      routeSubtreeHasSink_.assign(route.nodeStorage().size(), 0);
      routeSubtreeSlots_.push_back(*sourceSlot);
      for (std::size_t cursor = 0; cursor != routeSubtreeSlots_.size();
           ++cursor) {
        const PnrIndex slot = routeSubtreeSlots_[cursor];
        if (slot >= route.nodeStorage().size() ||
            !route.nodeStorage()[slot].isActive())
          return invalid("routed tree traversal reached an inactive node");
        for (PnrIndex child = route.nodeStorage()[slot].firstChild;
             child != getInvalidPnrIndex();
             child = route.nodeStorage()[child].nextSibling) {
          if (child >= route.nodeStorage().size())
            return invalid("routed tree child is out of range");
          routeSubtreeSlots_.push_back(child);
        }
      }
      for (auto slot = routeSubtreeSlots_.rbegin();
           slot != routeSubtreeSlots_.rend(); ++slot) {
        const RouteTreeNode &node = route.nodeStorage()[*slot];
        bool hasSink = node.sinkObligationCount != 0;
        for (PnrIndex child = node.firstChild; child != getInvalidPnrIndex();
             child = route.nodeStorage()[child].nextSibling)
          hasSink |= routeSubtreeHasSink_[child] != 0;
        routeSubtreeHasSink_[*slot] = hasSink;
        if (hasSink && node.endpoint != source)
          routeRootEndpoints_.push_back(node.endpoint);
      }
      llvm::sort(routeRootEndpoints_);
      for (PnrIndex endpoint : routeRootEndpoints_)
        transportChoices_.emplace_back(
            SpatialRootedSubtreeRoutingAction{logicalNet, endpoint});
    }
    if (llvm::Error error = appendTransportRange(offset))
      return error;
  }

  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet) {
    if (candidate.usesRegisterFifo(logicalNet))
      continue;
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
  const PnrIndex capacityDomainOffset = static_cast<PnrIndex>(
      preparedProblem_->resources().capacityDimensions().size());
  for (PnrIndex domain = 0;
       domain <
       preparedProblem_->routing().tagContinuity().matchDomains().size();
       ++domain) {
    if (candidate.tagDomainResidentCapacityOveruse(domain) == 0)
      continue;
    auto witness =
        checkedPnrIndexAdd({"SpatialActionDomain", "capacityWitnesses",
                            "Action", PnrCapacityMeasure::Index},
                           capacityDomainOffset, domain);
    if (!witness)
      return witness.takeError();
    if (llvm::Error error =
            appendWitness(ResolvedPnrViolationKind::CapacityOveruse, *witness))
      return error;
  }

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

  std::uint64_t hardProgressWitnessCount = 0;
  if (llvm::Error error =
          candidate.progress().enumerateFiniteBufferConflictOwners(
              hardProgressWitnessOwners_))
    return error;
  for (PnrIndex owner : hardProgressWitnessOwners_) {
    if (llvm::Error error = appendWitness(
            ResolvedPnrViolationKind::HardProgressViolation, owner))
      return error;
    if (hardProgressWitnessCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("hard-progress witness count overflows u64");
    ++hardProgressWitnessCount;
  }
  if (hardProgressWitnessCount >
      std::numeric_limits<std::uint64_t>::max() - externalNetCount)
    return invalid("transport movable decision count overflows u64");
  transportMovableDecisionCount_ = externalNetCount + hardProgressWitnessCount;

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
    if (resourceMovableDecisionCount_ ==
        std::numeric_limits<std::uint64_t>::max())
      return invalid("movable decision count overflows u64");
    ++resourceMovableDecisionCount_;
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

  const auto &memory = preparedProblem_->memory();
  for (PnrIndex binding = 0; binding < memory.logicalBindings().size();
       ++binding) {
    const std::size_t offset = resourceChoices_.size();
    auto generated =
        preparedProblem_->memoryConstraints().collectLogicalBindingChoices(
            binding, candidate.logicalMemoryBindings_, logicalMemoryChoices_);
    if (!generated)
      return generated.takeError();
    const SpatialLogicalMemoryBindingSelection &current =
        candidate.logicalMemoryBinding(binding);
    for (const SpatialLogicalMemoryBindingSelection &choice :
         llvm::ArrayRef(logicalMemoryChoices_).take_front(*generated)) {
      if (choice.target == current.target &&
          choice.physicalOffsetBytes == current.physicalOffsetBytes)
        continue;
      auto supported =
          candidate.logicalMemoryBindingTargetSupported(binding, choice.target);
      if (!supported)
        return supported.takeError();
      if (!*supported)
        continue;
      const std::array<PnrIndex, 1> fixedBindings{binding};
      const std::array<SpatialLogicalMemoryBindingSelection, 1> fixedSelections{
          choice};
      auto closed = preparedProblem_->memoryConstraints().solveCanonicalClosure(
          candidate.logicalMemoryBindings_, fixedBindings, fixedSelections,
          preparedProblem_->config()
              .policy()
              .search.initializer.assignmentAttemptLimitPerSeed,
          [&](PnrIndex dependentBinding,
              PnrIndex target) -> llvm::Expected<bool> {
            return candidate.logicalMemoryBindingTargetSupported(
                dependentBinding, target);
          },
          *memoryConstraintScratch_);
      bool publish = false;
      if (!closed) {
        bool workLimit = false;
        llvm::Error unhandled = llvm::handleErrors(
            closed.takeError(),
            [&](const detail::SpatialMemoryConstraintSolveFailure &)
                -> llvm::Error {
              workLimit = true;
              return llvm::Error::success();
            });
        if (unhandled)
          return unhandled;
        publish = workLimit;
      } else {
        publish = *closed;
      }
      if (publish)
        resourceChoices_.emplace_back(SpatialLogicalMemoryBindingAction{
            binding, choice.target, choice.physicalOffsetBytes});
    }
    if (llvm::Error error = appendResourceRange(offset))
      return error;
  }
  for (PnrIndex use = 0; use < memory.rootedUses().size(); ++use) {
    const std::size_t offset = resourceChoices_.size();
    auto domain = candidate.memoryDispatchDomain(use);
    if (!domain)
      return domain.takeError();
    for (PnrIndex option = (*domain)->optionOffset;
         option < (*domain)->optionOffset + (*domain)->optionCount; ++option) {
      if (candidate.memoryUseDispatch(use) == option)
        continue;
      auto supported =
          candidate.memoryUseDispatchSelectionSupported(use, option);
      if (!supported)
        return supported.takeError();
      if (!*supported)
        continue;
      resourceChoices_.emplace_back(
          SpatialMemoryUseDispatchAction{use, option});
    }
    if (llvm::Error error = appendResourceRange(offset))
      return error;
  }
  for (PnrIndex exposure = 0; exposure < memory.exposures().size();
       ++exposure) {
    const std::size_t offset = resourceChoices_.size();
    for (PnrIndex option = 0; option < memory.exposureOptions().size();
         ++option) {
      if (candidate.memoryExposureSelection(exposure) == option)
        continue;
      const PnrIndex binding = memory.exposures()[exposure].logicalBinding;
      if (!detail::memoryExposureMatchesTarget(
              memory.bindingTargets()[candidate.logicalMemoryBinding(binding)
                                          .target],
              memory.exposureOptions()[option]))
        continue;
      resourceChoices_.emplace_back(
          SpatialMemoryExposureAction{exposure, option});
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

llvm::Expected<std::optional<SpatialMappingAction>>
SpatialActionDomainScratch::propose(
    const ResolvedPnrActionProposalPolicy &policy,
    DeterministicPnrRandomStream &proposalStream) const {
  return detail::proposeCanonicalSpatialAction(policy, view(), proposalStream);
}

std::uint64_t SpatialActionDomainScratch::movableDecisionCount() const {
  return realizationMovableDecisionCount_ + transportMovableDecisionCount_ +
         resourceMovableDecisionCount_;
}

std::uint64_t SpatialActionDomainScratch::selectableMovableDecisionCount(
    const ResolvedPnrActionProposalPolicy &policy) const {
  return (policy.realizationBindingWeight != 0
              ? realizationMovableDecisionCount_
              : 0) +
         (policy.transportRoutingWeight != 0 ? transportMovableDecisionCount_
                                             : 0) +
         (policy.resourceAllocationWeight != 0 ? resourceMovableDecisionCount_
                                               : 0);
}

std::size_t SpatialActionDomainScratch::retainedStorageBytes() const {
  return retainedBytes(realizationAnchors_) +
         retainedBytes(realizationChoices_) + retainedBytes(transportAnchors_) +
         retainedBytes(transportChoices_) + retainedBytes(resourceAnchors_) +
         retainedBytes(resourceChoices_) + retainedBytes(relationChoices_) +
         retainedBytes(relationValueOffsets_) + retainedBytes(relationValues_) +
         retainedBytes(relationValueLoads_) +
         retainedBytes(relationDistinctValueCounts_) +
         retainedBytes(rootClosedRelations_) +
         retainedBytes(relationDecisionMemberOffsets_) +
         retainedBytes(relationDecisionMembers_) +
         retainedBytes(logicalMemoryChoices_) +
         retainedBytes(routeRootEndpoints_) +
         retainedBytes(routeSubtreeSlots_) +
         retainedBytes(routeSubtreeHasSink_) +
         retainedBytes(hardProgressWitnessOwners_) +
         (memoryConstraintScratch_
              ? memoryConstraintScratch_->retainedStorageBytes()
              : 0);
}
