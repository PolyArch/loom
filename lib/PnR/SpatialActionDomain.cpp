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
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cstdlib>
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
  progressShortfallWitnessOwners_.clear();
  progressDebtWitnessOwners_.clear();
  resourceAnchors_.clear();
  resourceChoices_.clear();
  realizationMovableDecisionCount_ = 0;
  transportMovableDecisionCount_ = 0;
  resourceMovableDecisionCount_ = 0;
  realizationAnchors_.reserve(realizationAnchorCapacity);
  realizationChoices_.reserve(realizationChoiceCapacity);
  previousRealizationChoices_.reserve(realizationChoiceCapacity);
  transportAnchors_.reserve(*transportAnchorCapacity);
  transportChoices_.reserve(*transportChoiceCapacity);
  previousTransportChoices_.reserve(*transportChoiceCapacity);
  realizationSegments_.resize(realizationAnchorCapacity);
  previousRealizationSegments_.resize(realizationAnchorCapacity);
  transportNetSegments_.resize(logicalNetCount);
  previousTransportNetSegments_.resize(logicalNetCount);
  sortedTouchedNets_.reserve(logicalNetCount);
  routeRootEndpoints_.reserve(endpointCount);
  routeSubtreeSlots_.reserve(endpointCount);
  routeSubtreeHasSink_.reserve(endpointCount);
  progressShortfallWitnessOwners_.reserve(
      problem.progressIndex().finiteBufferOwners().size());
  progressDebtWitnessOwners_.reserve(
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
  relationDecisionMemberChoiceValueOrdinals_.clear();
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
      const auto values =
          llvm::ArrayRef(relationValues_)
              .slice(relationValueOffsets_[relationOrdinal],
                     relationValueOffsets_[relationOrdinal + 1] -
                         relationValueOffsets_[relationOrdinal]);
      for (const detail::InitializerRelationMember &member :
           relations.relations().members(
               relations.relations().relations()[relationOrdinal]))
        if (member.decision == decision) {
          const std::size_t valueOffset =
              relationDecisionMemberChoiceValueOrdinals_.size();
          const PnrIndex choiceCount = decisionChoiceOffsets[decision + 1] -
                                       decisionChoiceOffsets[decision];
          for (PnrIndex choice = 0; choice < choiceCount; ++choice) {
            const PnrIndex rawValue =
                relations.relations().projectedValue(member, choice);
            const auto found = llvm::lower_bound(values, rawValue);
            if (found == values.end() || *found != rawValue)
              return invalid("binding relation value index is incomplete");
            auto value =
                actionIndex(static_cast<std::size_t>(found - values.begin()),
                            "relationValues", PnrCapacityMeasure::Index);
            if (!value)
              return value.takeError();
            relationDecisionMemberChoiceValueOrdinals_.push_back(*value);
          }
          relationDecisionMembers_.push_back({valueOffset, member.demand});
        }
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
  realizationAffectedMarks_.assign(relations.realizationDecisionCount(), 0);
  affectedEpoch_ = 0;
  segmentsValid_ = false;
  previousRealizationChoices_.clear();
  previousTransportChoices_.clear();
  touchedRelationValues_.reserve(relationDecisionMembers_.size());
  touchedEqualRelations_.reserve(relations.relations().relations().size());
  // Reverse incidence for incremental apply: which realization decisions can
  // change legality when a constraint or root-closed relation value's load
  // changes, and which realization decisions an equal-kind relation couples.
  {
    std::vector<std::pair<PnrIndex, PnrIndex>> valuePairs;
    std::vector<std::pair<PnrIndex, PnrIndex>> relationPairs;
    const auto incidenceOffsets =
        relations.relations().decisionRelationOffsets();
    for (PnrIndex decision = 0; decision < relations.realizationDecisionCount();
         ++decision) {
      const PnrIndex choiceCount =
          decisionChoiceOffsets[decision + 1] - decisionChoiceOffsets[decision];
      for (auto incidenceRecord :
           llvm::enumerate(relations.decisionRelations(decision))) {
        const PnrIndex relationOrdinal = incidenceRecord.value();
        if (!relations.relationIsConstraint(relationOrdinal) &&
            !rootClosedRelations_[relationOrdinal])
          continue;
        relationPairs.push_back({relationOrdinal, decision});
        const std::size_t incidence =
            incidenceOffsets[decision] + incidenceRecord.index();
        const std::size_t base = relationValueOffsets_[relationOrdinal];
        for (const RelationDecisionMember &record :
             llvm::ArrayRef(relationDecisionMembers_)
                 .slice(relationDecisionMemberOffsets_[incidence],
                        relationDecisionMemberOffsets_[incidence + 1] -
                            relationDecisionMemberOffsets_[incidence]))
          for (PnrIndex choice = 0; choice < choiceCount; ++choice) {
            const std::size_t value =
                base + relationDecisionMemberChoiceValueOrdinals_
                           [record.choiceValueOrdinalOffset + choice];
            auto checkedValue =
                actionIndex(value, "relationValues", PnrCapacityMeasure::Index);
            if (!checkedValue)
              return checkedValue.takeError();
            valuePairs.push_back({*checkedValue, decision});
          }
      }
    }
    llvm::sort(valuePairs);
    valuePairs.erase(std::unique(valuePairs.begin(), valuePairs.end()),
                     valuePairs.end());
    llvm::sort(relationPairs);
    relationPairs.erase(std::unique(relationPairs.begin(), relationPairs.end()),
                        relationPairs.end());
    valueRealizationOffsets_.assign(relationValues_.size() + 1, 0);
    valueRealizationDecisions_.clear();
    valueRealizationDecisions_.reserve(valuePairs.size());
    for (const auto &pair : valuePairs) {
      ++valueRealizationOffsets_[pair.first + 1];
      valueRealizationDecisions_.push_back(pair.second);
    }
    for (std::size_t value = 0; value < relationValues_.size(); ++value)
      valueRealizationOffsets_[value + 1] += valueRealizationOffsets_[value];
    relationRealizationOffsets_.assign(
        relations.relations().relations().size() + 1, 0);
    relationRealizationDecisions_.clear();
    relationRealizationDecisions_.reserve(relationPairs.size());
    for (const auto &pair : relationPairs) {
      ++relationRealizationOffsets_[pair.first + 1];
      relationRealizationDecisions_.push_back(pair.second);
    }
    for (std::size_t relation = 0;
         relation < relations.relations().relations().size(); ++relation)
      relationRealizationOffsets_[relation + 1] +=
          relationRealizationOffsets_[relation];
  }
  preparedProblem_ = &problem;
  return llvm::Error::success();
}

llvm::Error
SpatialActionDomainScratch::appendRealizationRange(std::size_t offset) {
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
}

llvm::Error
SpatialActionDomainScratch::appendTransportRange(std::size_t offset) {
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
}

llvm::Error
SpatialActionDomainScratch::appendResourceRange(std::size_t offset) {
  if (resourceChoices_.size() == offset)
    return llvm::Error::success();
  auto checkedOffset =
      actionIndex(offset, "resourceChoices", PnrCapacityMeasure::Offset);
  if (!checkedOffset)
    return checkedOffset.takeError();
  auto checkedCount = actionIndex(resourceChoices_.size() - offset,
                                  "resourceChoices", PnrCapacityMeasure::Count);
  if (!checkedCount)
    return checkedCount.takeError();
  resourceAnchors_.push_back({*checkedOffset, *checkedCount});
  if (resourceMovableDecisionCount_ ==
      std::numeric_limits<std::uint64_t>::max())
    return invalid("movable decision count overflows u64");
  ++resourceMovableDecisionCount_;
  return llvm::Error::success();
}

bool SpatialActionDomainScratch::relationChoiceIsLegal(
    PnrIndex decision, PnrIndex localChoice, bool constraintsOnly) const {
  const detail::SpatialBindingRelationModel &relations =
      preparedProblem_->bindingRelations();
  const detail::InitializerRelationModel &relationModel = relations.relations();
  const PnrIndex oldChoice = relationChoices_[decision];
  const auto decisionRelationOffsets = relationModel.decisionRelationOffsets();
  for (auto incidenceRecord :
       llvm::enumerate(relations.decisionRelations(decision))) {
    const std::size_t localIncidence = incidenceRecord.index();
    const PnrIndex relationOrdinal = incidenceRecord.value();
    if (constraintsOnly && !relations.relationIsConstraint(relationOrdinal) &&
        !rootClosedRelations_[relationOrdinal])
      continue;
    const detail::InitializerRelationRecord &relation =
        relationModel.relations()[relationOrdinal];
    auto loads = llvm::MutableArrayRef(const_cast<std::vector<std::uint64_t> &>(
                                           relationValueLoads_))
                     .slice(relationValueOffsets_[relationOrdinal],
                            relationValueOffsets_[relationOrdinal + 1] -
                                relationValueOffsets_[relationOrdinal]);
    auto &distinctValueCounts =
        const_cast<std::vector<PnrIndex> &>(relationDistinctValueCounts_);
    const auto capacities = relationModel.valueCapacities(relation);
    const std::size_t incidence =
        decisionRelationOffsets[decision] + localIncidence;
    const auto changedMembers =
        llvm::ArrayRef(relationDecisionMembers_)
            .slice(relationDecisionMemberOffsets_[incidence],
                   relationDecisionMemberOffsets_[incidence + 1] -
                       relationDecisionMemberOffsets_[incidence]);
    const auto memberValue = [&](const RelationDecisionMember &record,
                                 PnrIndex choice) {
      const std::size_t offset = record.choiceValueOrdinalOffset + choice;
      assert(offset < relationDecisionMemberChoiceValueOrdinals_.size());
      return relationDecisionMemberChoiceValueOrdinals_[offset];
    };
    if (llvm::all_of(changedMembers, [&](const RelationDecisionMember &record) {
          return memberValue(record, oldChoice) ==
                 memberValue(record, localChoice);
        }))
      continue;
    const auto update = [&](PnrIndex choice, bool add) {
      bool legal = true;
      for (const RelationDecisionMember &record : changedMembers) {
        const PnrIndex value = memberValue(record, choice);
        const std::uint64_t demand =
            relation.kind == detail::InitializerRelationKind::Capacity
                ? record.demand
                : 1;
        if (add) {
          assert(demand <=
                 std::numeric_limits<std::uint64_t>::max() - loads[value]);
          if (loads[value] == 0)
            ++distinctValueCounts[relationOrdinal];
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
            --distinctValueCounts[relationOrdinal];
        }
      }
      return legal;
    };
    update(oldChoice, false);
    bool legal = update(localChoice, true);
    if (relation.kind == detail::InitializerRelationKind::Equal &&
        distinctValueCounts[relationOrdinal] != 1)
      legal = false;
    update(localChoice, false);
    update(oldChoice, true);
    if (!legal)
      return false;
  }
  return true;
}

llvm::Error SpatialActionDomainScratch::rebuildRelationLoads(
    const SpatialCandidateState &candidate) {
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
  return llvm::Error::success();
}

llvm::Error SpatialActionDomainScratch::emitRealizationSegment(
    const SpatialCandidateState &candidate, PnrIndex decision) {
  const detail::SpatialBindingRelationModel &relations =
      preparedProblem_->bindingRelations();
  const std::size_t offset = realizationChoices_.size();
  const PnrIndex memoryDecisionOffset = relations.computeDecisionCount();
  if (decision < memoryDecisionOffset) {
    const PnrIndex realization = decision;
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
  } else {
    const PnrIndex realization = decision - memoryDecisionOffset;
    const auto choices = relations.memoryChoices(realization);
    for (auto [localChoice, choice] : llvm::enumerate(choices)) {
      if (candidate.memoryBinding(realization).placement == choice.placement)
        continue;
      ++examinedRealizationChoiceCount_;
      if (!relationChoiceIsLegal(decision, static_cast<PnrIndex>(localChoice),
                                 true)) {
        ++fixedRelationPrunedRealizationChoiceCount_;
        continue;
      }
      realizationChoices_.emplace_back(
          SpatialMemoryBindingAction{realization, choice.placement});
    }
  }
  auto checkedOffset =
      actionIndex(offset, "realizationSegments", PnrCapacityMeasure::Offset);
  if (!checkedOffset)
    return checkedOffset.takeError();
  auto checkedCount =
      actionIndex(realizationChoices_.size() - offset, "realizationSegments",
                  PnrCapacityMeasure::Count);
  if (!checkedCount)
    return checkedCount.takeError();
  realizationSegments_[decision] = {*checkedOffset, *checkedCount};
  return appendRealizationRange(offset);
}

llvm::Error SpatialActionDomainScratch::emitTransportNetSegment(
    const SpatialCandidateState &candidate, PnrIndex logicalNet) {
  const auto &transfers = preparedProblem_->transfers();
  if (candidate.usesRegisterFifo(logicalNet)) {
    transportNetSegments_[logicalNet] = {0, 0};
    return llvm::Error::success();
  }
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
  auto checkedOffset =
      actionIndex(offset, "transportNetSegments", PnrCapacityMeasure::Offset);
  if (!checkedOffset)
    return checkedOffset.takeError();
  auto checkedCount =
      actionIndex(transportChoices_.size() - offset, "transportNetSegments",
                  PnrCapacityMeasure::Count);
  if (!checkedCount)
    return checkedCount.takeError();
  transportNetSegments_[logicalNet] = {*checkedOffset, *checkedCount};
  return appendTransportRange(offset);
}

llvm::Error SpatialActionDomainScratch::emitTransportWitnessTail(
    const SpatialCandidateState &candidate, std::uint64_t externalNetCount) {
  const auto &transfers = preparedProblem_->transfers();
  const auto appendWitness = [&](ResolvedPnrViolationKind kind,
                                 PnrIndex ordinal) -> llvm::Error {
    const std::size_t offset = transportChoices_.size();
    transportChoices_.emplace_back(
        SpatialWitnessRegionRoutingAction{kind, ordinal});
    return appendTransportRange(offset);
  };
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

  std::uint64_t progressWitnessCount = 0;
  if (llvm::Error error =
          candidate.progress().enumerateCapacityShortfallOwners(
              progressShortfallWitnessOwners_))
    return error;
  for (PnrIndex owner : progressShortfallWitnessOwners_) {
    if (llvm::Error error = appendWitness(
            ResolvedPnrViolationKind::HardProgressViolation, owner))
      return error;
    if (progressWitnessCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("progress witness count overflows u64");
    ++progressWitnessCount;
  }
  if (llvm::Error error =
          candidate.progress().enumerateCapacityProofDebtOwners(
              progressDebtWitnessOwners_))
    return error;
  for (PnrIndex owner : progressDebtWitnessOwners_) {
    if (llvm::Error error = appendWitness(
            ResolvedPnrViolationKind::ProgressProofDebt, owner))
      return error;
    if (progressWitnessCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("progress witness count overflows u64");
    ++progressWitnessCount;
  }
  const auto noGoodClauses = preparedProblem_->constraints().resolvedNoGoods();
  for (PnrIndex clause = 0; clause < noGoodClauses.size(); ++clause) {
    if (!candidate.runtimeCounterexampleClauseViolated(clause))
      continue;
    if (llvm::Error error = appendWitness(
            ResolvedPnrViolationKind::RuntimeCounterexampleViolation, clause))
      return error;
    if (progressWitnessCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("transport witness count overflows u64");
    ++progressWitnessCount;
  }
  if (candidate.selectedHandshakeViolation() != 0) {
    if (candidate.selectedHandshakeRegisterFifoCut() == getInvalidPnrIndex())
      return invalid("selected handshake cycle has no register-FIFO cut");
    if (llvm::Error error = appendWitness(
            ResolvedPnrViolationKind::SelectedHandshakeViolation, 0))
      return error;
    if (progressWitnessCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("transport witness count overflows u64");
    ++progressWitnessCount;
  }
  if (progressWitnessCount >
      std::numeric_limits<std::uint64_t>::max() - externalNetCount)
    return invalid("transport movable decision count overflows u64");
  transportMovableDecisionCount_ = externalNetCount + progressWitnessCount;
  return llvm::Error::success();
}

llvm::Error SpatialActionDomainScratch::rebuildResourceSection(
    const SpatialCandidateState &candidate) {
  const detail::SpatialBindingRelationModel &relations =
      preparedProblem_->bindingRelations();
  const auto &realizations = preparedProblem_->realizations();
  resourceAnchors_.clear();
  resourceChoices_.clear();
  resourceMovableDecisionCount_ = 0;
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
  progressShortfallWitnessOwners_.clear();
  progressDebtWitnessOwners_.clear();
  realizationMovableDecisionCount_ = 0;
  transportMovableDecisionCount_ = 0;
  examinedRealizationChoiceCount_ = 0;
  fixedRelationPrunedRealizationChoiceCount_ = 0;

  if (llvm::Error error = rebuildRelationLoads(candidate))
    return error;

  const detail::SpatialBindingRelationModel &relations =
      preparedProblem_->bindingRelations();
  std::fill(realizationSegments_.begin(), realizationSegments_.end(),
            SpatialActionChoiceRange{0, 0});
  for (PnrIndex decision = 0; decision < relations.realizationDecisionCount();
       ++decision)
    if (llvm::Error error = emitRealizationSegment(candidate, decision))
      return error;

  const auto &transfers = preparedProblem_->transfers();
  std::fill(transportNetSegments_.begin(), transportNetSegments_.end(),
            SpatialActionChoiceRange{0, 0});
  std::uint64_t externalNetCount = 0;
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet) {
    if (llvm::Error error = emitTransportNetSegment(candidate, logicalNet))
      return error;
    externalNetCount += transportNetSegments_[logicalNet].choiceCount != 0;
  }
  if (llvm::Error error = emitTransportWitnessTail(candidate, externalNetCount))
    return error;

  if (llvm::Error error = rebuildResourceSection(candidate))
    return error;
  segmentsValid_ = true;
  return llvm::Error::success();
}

llvm::Error SpatialActionDomainScratch::applyCommitted(
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<std::pair<SpatialCandidateScratch::DecisionKind, PnrIndex>>
        changedDecisions,
    llvm::ArrayRef<PnrIndex> touchedLogicalNets) {
  if (preparedProblem_ == nullptr)
    return invalid("scratch storage was not prepared");
  if (&candidate.problem() != preparedProblem_)
    return invalid("candidate belongs to a different Frozen problem");
  using DecisionKind = SpatialCandidateScratch::DecisionKind;
  bool fallback = !segmentsValid_;
  for (const auto &change : changedDecisions)
    fallback |= change.first == DecisionKind::RegisterFifoTransfer;
  if (fallback)
    return rebuild(candidate);

  const detail::SpatialBindingRelationModel &relations =
      preparedProblem_->bindingRelations();
  const detail::InitializerRelationModel &relationModel = relations.relations();
  if (++affectedEpoch_ == 0) {
    std::fill(realizationAffectedMarks_.begin(),
              realizationAffectedMarks_.end(), 0);
    affectedEpoch_ = 1;
  }
  const std::uint64_t epoch = affectedEpoch_;
  const auto markValue = [&](std::size_t value) {
    for (std::size_t at = valueRealizationOffsets_[value];
         at < valueRealizationOffsets_[value + 1]; ++at)
      realizationAffectedMarks_[valueRealizationDecisions_[at]] = epoch;
  };
  const auto incidenceOffsets = relationModel.decisionRelationOffsets();
  // Committed state is only consistent after every changed decision's load
  // update lands, so relation validation runs over the touched values and
  // relations afterwards instead of on per-decision transients.
  touchedRelationValues_.clear();
  touchedEqualRelations_.clear();
  for (const auto &change : changedDecisions) {
    PnrIndex decision = 0;
    switch (change.first) {
    case DecisionKind::ComputeBinding:
      decision = change.second;
      break;
    case DecisionKind::MemoryBinding:
      decision = relations.computeDecisionCount() + change.second;
      break;
    case DecisionKind::PortAttachment:
      decision = relations.portDecisionOffset() + change.second;
      break;
    case DecisionKind::GraphBoundaryAttachment:
      decision = relations.graphBoundaryDecisionOffset() + change.second;
      break;
    case DecisionKind::MemoryOperationPlan:
    case DecisionKind::LogicalMemoryBinding:
    case DecisionKind::MemoryUseDispatch:
    case DecisionKind::MemoryExposure:
    case DecisionKind::RegisterFifoTransfer:
      continue;
    }
    if (decision >= relationChoices_.size())
      return invalid("changed decision is outside the relation domain");
    if (decision < relations.realizationDecisionCount())
      realizationAffectedMarks_[decision] = epoch;
    const PnrIndex oldChoice = relationChoices_[decision];
    const PnrIndex newChoice = candidate.bindingRelationChoices_[decision];
    if (oldChoice == newChoice)
      continue;
    for (auto incidenceRecord :
         llvm::enumerate(relations.decisionRelations(decision))) {
      const PnrIndex relationOrdinal = incidenceRecord.value();
      const detail::InitializerRelationRecord &relation =
          relationModel.relations()[relationOrdinal];
      const std::size_t incidence =
          incidenceOffsets[decision] + incidenceRecord.index();
      const bool gated = relations.relationIsConstraint(relationOrdinal) ||
                         rootClosedRelations_[relationOrdinal];
      const std::size_t base = relationValueOffsets_[relationOrdinal];
      bool valueChanged = false;
      for (const RelationDecisionMember &record :
           llvm::ArrayRef(relationDecisionMembers_)
               .slice(relationDecisionMemberOffsets_[incidence],
                      relationDecisionMemberOffsets_[incidence + 1] -
                          relationDecisionMemberOffsets_[incidence])) {
        const PnrIndex oldValue = relationDecisionMemberChoiceValueOrdinals_
            [record.choiceValueOrdinalOffset + oldChoice];
        const PnrIndex newValue = relationDecisionMemberChoiceValueOrdinals_
            [record.choiceValueOrdinalOffset + newChoice];
        if (oldValue == newValue)
          continue;
        valueChanged = true;
        const std::uint64_t demand =
            relation.kind == detail::InitializerRelationKind::Capacity
                ? record.demand
                : 1;
        std::uint64_t &oldLoad = relationValueLoads_[base + oldValue];
        if (oldLoad < demand)
          return invalid("binding relation load underflows");
        oldLoad -= demand;
        if (oldLoad == 0)
          --relationDistinctValueCounts_[relationOrdinal];
        std::uint64_t &newLoad = relationValueLoads_[base + newValue];
        if (demand > std::numeric_limits<std::uint64_t>::max() - newLoad)
          return invalid("binding relation load overflows u64");
        if (newLoad == 0)
          ++relationDistinctValueCounts_[relationOrdinal];
        newLoad += demand;
        touchedRelationValues_.push_back({relationOrdinal, newValue});
        if (gated) {
          markValue(base + oldValue);
          markValue(base + newValue);
        }
      }
      if (valueChanged) {
        if (relation.kind == detail::InitializerRelationKind::Equal) {
          touchedEqualRelations_.push_back(relationOrdinal);
          if (gated)
            for (std::size_t at = relationRealizationOffsets_[relationOrdinal];
                 at < relationRealizationOffsets_[relationOrdinal + 1]; ++at)
              realizationAffectedMarks_[relationRealizationDecisions_[at]] =
                  epoch;
        }
      }
    }
    relationChoices_[decision] = newChoice;
  }
  for (const auto &touched : touchedRelationValues_) {
    const detail::InitializerRelationRecord &relation =
        relationModel.relations()[touched.first];
    const std::uint64_t load =
        relationValueLoads_[relationValueOffsets_[touched.first] +
                            touched.second];
    if (relation.kind == detail::InitializerRelationKind::Disjoint && load > 1)
      return invalid("candidate violates a disjoint binding relation");
    if (relation.kind == detail::InitializerRelationKind::Capacity) {
      const auto capacities = relationModel.valueCapacities(relation);
      if (touched.second >= capacities.size() ||
          load > capacities[touched.second])
        return invalid("candidate violates a binding capacity relation");
    }
  }
  for (const PnrIndex relationOrdinal : touchedEqualRelations_)
    if (relationDistinctValueCounts_[relationOrdinal] != 1)
      return invalid("candidate violates an equal binding relation");

  std::swap(realizationChoices_, previousRealizationChoices_);
  std::swap(realizationSegments_, previousRealizationSegments_);
  realizationChoices_.clear();
  realizationAnchors_.clear();
  std::fill(realizationSegments_.begin(), realizationSegments_.end(),
            SpatialActionChoiceRange{0, 0});
  realizationMovableDecisionCount_ = 0;
  examinedRealizationChoiceCount_ = 0;
  fixedRelationPrunedRealizationChoiceCount_ = 0;
  for (PnrIndex decision = 0; decision < relations.realizationDecisionCount();
       ++decision) {
    if (realizationAffectedMarks_[decision] == epoch) {
      if (llvm::Error error = emitRealizationSegment(candidate, decision))
        return error;
      continue;
    }
    const SpatialActionChoiceRange segment =
        previousRealizationSegments_[decision];
    const std::size_t offset = realizationChoices_.size();
    realizationChoices_.insert(realizationChoices_.end(),
                               previousRealizationChoices_.begin() +
                                   segment.choiceOffset,
                               previousRealizationChoices_.begin() +
                                   segment.choiceOffset + segment.choiceCount);
    auto checkedOffset =
        actionIndex(offset, "realizationSegments", PnrCapacityMeasure::Offset);
    if (!checkedOffset)
      return checkedOffset.takeError();
    realizationSegments_[decision] = {*checkedOffset, segment.choiceCount};
    if (llvm::Error error = appendRealizationRange(offset))
      return error;
  }

  sortedTouchedNets_.assign(touchedLogicalNets.begin(),
                            touchedLogicalNets.end());
  llvm::sort(sortedTouchedNets_);
  std::swap(transportChoices_, previousTransportChoices_);
  std::swap(transportNetSegments_, previousTransportNetSegments_);
  transportChoices_.clear();
  transportAnchors_.clear();
  const auto &transfers = preparedProblem_->transfers();
  std::fill(transportNetSegments_.begin(), transportNetSegments_.end(),
            SpatialActionChoiceRange{0, 0});
  std::uint64_t externalNetCount = 0;
  for (PnrIndex logicalNet = 0; logicalNet < transfers.logicalNets().size();
       ++logicalNet) {
    if (std::binary_search(sortedTouchedNets_.begin(), sortedTouchedNets_.end(),
                           logicalNet)) {
      if (llvm::Error error = emitTransportNetSegment(candidate, logicalNet))
        return error;
    } else {
      const SpatialActionChoiceRange segment =
          previousTransportNetSegments_[logicalNet];
      const std::size_t offset = transportChoices_.size();
      transportChoices_.insert(transportChoices_.end(),
                               previousTransportChoices_.begin() +
                                   segment.choiceOffset,
                               previousTransportChoices_.begin() +
                                   segment.choiceOffset + segment.choiceCount);
      auto checkedOffset = actionIndex(offset, "transportNetSegments",
                                       PnrCapacityMeasure::Offset);
      if (!checkedOffset)
        return checkedOffset.takeError();
      transportNetSegments_[logicalNet] = {*checkedOffset, segment.choiceCount};
      if (llvm::Error error = appendTransportRange(offset))
        return error;
    }
    externalNetCount += transportNetSegments_[logicalNet].choiceCount != 0;
  }
  if (llvm::Error error = emitTransportWitnessTail(candidate, externalNetCount))
    return error;

  if (llvm::Error error = rebuildResourceSection(candidate))
    return error;

  static const bool verifyAgainstRebuild = [] {
    const char *value = std::getenv("LOOM_PNR_VERIFY_ACTION_DOMAIN");
    return value != nullptr && value[0] != '0';
  }();
  if (verifyAgainstRebuild) {
    const auto appliedRealizationAnchors = realizationAnchors_;
    const auto appliedRealizationChoices = realizationChoices_;
    const auto appliedTransportAnchors = transportAnchors_;
    const auto appliedTransportChoices = transportChoices_;
    const auto appliedResourceAnchors = resourceAnchors_;
    const auto appliedResourceChoices = resourceChoices_;
    const auto appliedLoads = relationValueLoads_;
    if (llvm::Error error = rebuild(candidate))
      return error;
    const auto sameAnchors = [](const auto &lhs, const auto &rhs) {
      if (lhs.size() != rhs.size())
        return false;
      for (std::size_t at = 0; at < lhs.size(); ++at)
        if (lhs[at].choiceOffset != rhs[at].choiceOffset ||
            lhs[at].choiceCount != rhs[at].choiceCount)
          return false;
      return true;
    };
    const auto sameChoices = [](const auto &lhs, const auto &rhs) {
      if (lhs.size() != rhs.size())
        return false;
      for (std::size_t at = 0; at < lhs.size(); ++at)
        if (!(spatialActionKey(SpatialMappingAction(lhs[at])) ==
              spatialActionKey(SpatialMappingAction(rhs[at]))))
          return false;
      return true;
    };
    if (!sameAnchors(appliedRealizationAnchors, realizationAnchors_) ||
        !sameChoices(appliedRealizationChoices, realizationChoices_) ||
        !sameAnchors(appliedTransportAnchors, transportAnchors_) ||
        !sameChoices(appliedTransportChoices, transportChoices_) ||
        !sameAnchors(appliedResourceAnchors, resourceAnchors_) ||
        !sameChoices(appliedResourceChoices, resourceChoices_) ||
        appliedLoads != relationValueLoads_)
      return invalid("incremental Action domain diverged from a full rebuild");
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
         retainedBytes(previousRealizationChoices_) +
         retainedBytes(previousTransportChoices_) +
         retainedBytes(realizationSegments_) +
         retainedBytes(previousRealizationSegments_) +
         retainedBytes(transportNetSegments_) +
         retainedBytes(previousTransportNetSegments_) +
         retainedBytes(valueRealizationOffsets_) +
         retainedBytes(valueRealizationDecisions_) +
         retainedBytes(relationRealizationOffsets_) +
         retainedBytes(relationRealizationDecisions_) +
         retainedBytes(realizationAffectedMarks_) +
         retainedBytes(touchedRelationValues_) +
         retainedBytes(touchedEqualRelations_) +
         retainedBytes(sortedTouchedNets_) + retainedBytes(transportChoices_) +
         retainedBytes(resourceAnchors_) + retainedBytes(resourceChoices_) +
         retainedBytes(relationChoices_) +
         retainedBytes(relationValueOffsets_) + retainedBytes(relationValues_) +
         retainedBytes(relationValueLoads_) +
         retainedBytes(relationDistinctValueCounts_) +
         retainedBytes(rootClosedRelations_) +
         retainedBytes(relationDecisionMemberOffsets_) +
         retainedBytes(relationDecisionMembers_) +
         retainedBytes(relationDecisionMemberChoiceValueOrdinals_) +
         retainedBytes(logicalMemoryChoices_) +
         retainedBytes(routeRootEndpoints_) +
         retainedBytes(routeSubtreeSlots_) +
         retainedBytes(routeSubtreeHasSink_) +
         retainedBytes(progressShortfallWitnessOwners_) +
         retainedBytes(progressDebtWitnessOwners_) +
         (memoryConstraintScratch_
              ? memoryConstraintScratch_->retainedStorageBytes()
              : 0);
}
