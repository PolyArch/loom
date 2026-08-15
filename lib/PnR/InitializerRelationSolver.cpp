#include "InitializerRelationSolver.h"

#include "Common/MappingDebugLog.h"
#include "InitializerChoiceOrder.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <system_error>

using namespace loom::pnr;
using namespace loom::pnr::detail;

char InitializerRelationSolveFailure::ID;

void InitializerRelationSolveFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code InitializerRelationSolveFailure::convertToErrorCode() const {
  return std::make_error_code(
      kind_ == InitializerRelationSolveFailureKind::WorkLimit
          ? std::errc::resource_unavailable_try_again
          : std::errc::invalid_argument);
}

namespace {

llvm::Error modelError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid initializer relation model: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

llvm::Error assignmentError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid initializer relation assignment: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

llvm::Expected<InitializerRelationModel> InitializerRelationModel::create(
    std::vector<PnrIndex> decisionChoiceCounts,
    std::vector<InitializerRelationInput> relationInputs) {
  InitializerRelationModel model;
  if (decisionChoiceCounts.size() > getPnrIndexMax())
    return modelError("decision count overflows PnrIndex");
  model.decisionChoiceOffsets_.reserve(decisionChoiceCounts.size() + 1);
  PnrIndex choiceOffset = 0;
  model.decisionChoiceOffsets_.push_back(choiceOffset);
  for (PnrIndex count : decisionChoiceCounts) {
    if (count == 0)
      return modelError("a decision has an empty choice domain");
    if (count > getPnrIndexMax() - choiceOffset)
      return modelError("the flattened choice domain overflows PnrIndex");
    choiceOffset += count;
    model.decisionChoiceOffsets_.push_back(choiceOffset);
  }

  const PnrIndex decisionCount =
      static_cast<PnrIndex>(decisionChoiceCounts.size());
  if (relationInputs.size() > getPnrIndexMax())
    return modelError("relation count overflows PnrIndex");
  std::vector<std::vector<PnrIndex>> decisionRelations(decisionCount);
  model.relations_.reserve(relationInputs.size());
  for (PnrIndex relationOrdinal = 0; relationOrdinal < relationInputs.size();
       ++relationOrdinal) {
    InitializerRelationInput &input = relationInputs[relationOrdinal];
    if (input.members.empty() ||
        (input.kind != InitializerRelationKind::Capacity &&
         input.members.size() < 2))
      return modelError("a relation has fewer than two members");
    if (input.kind == InitializerRelationKind::Capacity) {
      if (input.valueCapacities.empty())
        return modelError("a capacity relation has no value capacities");
      if (input.valueCapacities.size() >
          getPnrIndexMax() - model.valueCapacities_.size())
        return modelError("relation value capacities overflow PnrIndex");
      if (llvm::any_of(input.valueCapacities, [](std::uint64_t capacity) {
            return capacity > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::int64_t>::max());
          }))
        return modelError(
            "a relation value capacity is outside the nonnegative i64 domain");
      std::vector<PnrIndex> decisions;
      decisions.reserve(input.members.size());
      for (const InitializerRelationMemberInput &member : input.members)
        decisions.push_back(member.decision);
      llvm::sort(decisions);
      if (std::adjacent_find(decisions.begin(), decisions.end()) !=
          decisions.end())
        return modelError(
            "a capacity relation repeats one decision as a member");
    } else if (!input.valueCapacities.empty()) {
      return modelError("a non-capacity relation carries value capacities");
    }
    if (input.members.size() > getPnrIndexMax() - model.relationMembers_.size())
      return modelError("relation member count overflows PnrIndex");
    const PnrIndex memberOffset =
        static_cast<PnrIndex>(model.relationMembers_.size());
    const PnrIndex valueCapacityOffset =
        static_cast<PnrIndex>(model.valueCapacities_.size());
    for (InitializerRelationMemberInput &member : input.members) {
      if (member.decision >= decisionCount)
        return modelError("a relation member names a foreign decision");
      if (member.demand == 0 ||
          member.demand > static_cast<std::uint64_t>(
                              std::numeric_limits<std::int64_t>::max()))
        return modelError(
            "a relation member demand is outside the positive i64 domain");
      if (input.kind != InitializerRelationKind::Capacity && member.demand != 1)
        return modelError("a non-capacity relation carries member demand");
      if (member.projectedValues.size() !=
          decisionChoiceCounts[member.decision])
        return modelError(
            "a relation projection does not cover its decision domain");
      if (input.kind == InitializerRelationKind::Capacity &&
          llvm::any_of(member.projectedValues, [&](PnrIndex value) {
            return value >= input.valueCapacities.size();
          }))
        return modelError(
            "a capacity projection names a value without capacity");
      if (member.projectedValues.size() >
          getPnrIndexMax() - model.projectedValues_.size())
        return modelError("projected relation values overflow PnrIndex");
      const PnrIndex projectedValueOffset =
          static_cast<PnrIndex>(model.projectedValues_.size());
      model.projectedValues_.insert(model.projectedValues_.end(),
                                    member.projectedValues.begin(),
                                    member.projectedValues.end());
      model.relationMembers_.push_back(
          {member.decision, projectedValueOffset, member.demand});
      decisionRelations[member.decision].push_back(relationOrdinal);
    }
    model.valueCapacities_.insert(model.valueCapacities_.end(),
                                  input.valueCapacities.begin(),
                                  input.valueCapacities.end());
    model.relations_.push_back(
        {input.kind, memberOffset, static_cast<PnrIndex>(input.members.size()),
         valueCapacityOffset,
         static_cast<PnrIndex>(input.valueCapacities.size())});
  }

  model.decisionRelationOffsets_.reserve(decisionCount + 1);
  model.decisionRelationOffsets_.push_back(0);
  for (std::vector<PnrIndex> &incidence : decisionRelations) {
    llvm::sort(incidence);
    incidence.erase(std::unique(incidence.begin(), incidence.end()),
                    incidence.end());
    if (incidence.size() > getPnrIndexMax() - model.decisionRelations_.size())
      return modelError("decision relation incidence overflows PnrIndex");
    model.decisionRelations_.insert(model.decisionRelations_.end(),
                                    incidence.begin(), incidence.end());
    model.decisionRelationOffsets_.push_back(
        static_cast<PnrIndex>(model.decisionRelations_.size()));
  }
  return model;
}

bool InitializerRelationModel::relationSatisfied(
    PnrIndex relation, llvm::ArrayRef<PnrIndex> choices) const {
  assert(relation < relations_.size());
  assert(choices.size() == decisionCount());
  const InitializerRelationRecord &record = relations_[relation];
  const auto relationMembers = members(record);
  if (record.kind == InitializerRelationKind::Capacity) {
    std::vector<std::uint64_t> loads(record.valueCapacityCount, 0);
    const auto capacities = valueCapacities(record);
    for (const InitializerRelationMember &member : relationMembers) {
      assert(choices[member.decision] <
             decisionChoiceOffsets_[member.decision + 1] -
                 decisionChoiceOffsets_[member.decision]);
      const PnrIndex value = projectedValue(member, choices[member.decision]);
      assert(value < capacities.size());
      if (member.demand > capacities[value] - loads[value])
        return false;
      loads[value] += member.demand;
    }
    return true;
  }
  std::optional<PnrIndex> equalValue;
  for (std::size_t lhs = 0; lhs < relationMembers.size(); ++lhs) {
    const InitializerRelationMember &member = relationMembers[lhs];
    assert(choices[member.decision] <
           decisionChoiceOffsets_[member.decision + 1] -
               decisionChoiceOffsets_[member.decision]);
    const PnrIndex value = projectedValue(member, choices[member.decision]);
    if (record.kind == InitializerRelationKind::Equal) {
      if (equalValue && *equalValue != value)
        return false;
      equalValue = value;
      continue;
    }
    for (std::size_t rhs = 0; rhs < lhs; ++rhs) {
      const InitializerRelationMember &other = relationMembers[rhs];
      if (projectedValue(other, choices[other.decision]) == value)
        return false;
    }
  }
  return true;
}

llvm::Error InitializerRelationModel::verifyChoices(
    llvm::ArrayRef<PnrIndex> choices) const {
  if (choices.size() != decisionCount())
    return assignmentError("choice count does not match the decision domain");
  for (PnrIndex decision = 0; decision < choices.size(); ++decision)
    if (choices[decision] >=
        decisionChoiceOffsets_[decision + 1] - decisionChoiceOffsets_[decision])
      return assignmentError("a choice is outside its decision domain");
  for (PnrIndex relation = 0; relation < relations_.size(); ++relation)
    if (!relationSatisfied(relation, choices))
      return assignmentError("a hard relation failed");
  return llvm::Error::success();
}

InitializerRelationSolver::InitializerRelationSolver(
    const InitializerRelationModel &model,
    llvm::ArrayRef<PnrIndex> independentChoiceCounts)
    : model_(&model), decisionChoiceOffsets_(model.decisionChoiceOffsets()) {
  assert(independentChoiceCounts.size() <=
         getPnrIndexMax() - model.decisionCount());
  decisionChoiceOffsets_.reserve(decisionChoiceOffsets_.size() +
                                 independentChoiceCounts.size());
  PnrIndex choiceCount = decisionChoiceOffsets_.back();
  for (PnrIndex count : independentChoiceCounts) {
    assert(count != 0 && count <= getPnrIndexMax() - choiceCount);
    choiceCount += count;
    decisionChoiceOffsets_.push_back(choiceCount);
  }
  activeChoices_.resize(choiceCount);
  domainCounts_.resize(decisionChoiceOffsets_.size() - 1);
  removalJournal_.reserve(choiceCount);
  relationQueue_.resize(model.relations().size());
  relationPending_.resize(model.relations().size());
  binaryEqualSupports_.resize(model.relations().size());
  allDifferentSupports_.resize(model.relations().size());
  for (PnrIndex relation = 0; relation < model.relations().size(); ++relation) {
    const InitializerRelationRecord &record = model.relations()[relation];
    const auto members = model.members(record);
    if (record.kind == InitializerRelationKind::Equal && members.size() == 2 &&
        members[0].decision != members[1].decision) {
      BinaryEqualSupport &support = binaryEqualSupports_[relation];
      support.first = members[0];
      support.second = members[1];
      std::vector<PnrIndex> values;
      for (const InitializerRelationMember &member : members) {
        const PnrIndex choiceCount =
            decisionChoiceOffsets_[member.decision + 1] -
            decisionChoiceOffsets_[member.decision];
        values.reserve(values.size() + choiceCount);
        for (PnrIndex choice = 0; choice < choiceCount; ++choice)
          values.push_back(model.projectedValue(member, choice));
      }
      llvm::sort(values);
      values.erase(std::unique(values.begin(), values.end()), values.end());
      assert(!values.empty() && values.size() <= getPnrIndexMax());
      support.valueCount = static_cast<PnrIndex>(values.size());

      const auto appendChoiceValues =
          [&](const InitializerRelationMember &member) {
            const std::size_t offset = binaryEqualChoiceValues_.size();
            const PnrIndex choiceCount =
                decisionChoiceOffsets_[member.decision + 1] -
                decisionChoiceOffsets_[member.decision];
            for (PnrIndex choice = 0; choice < choiceCount; ++choice) {
              const PnrIndex projected = model.projectedValue(member, choice);
              const auto found = llvm::lower_bound(values, projected);
              assert(found != values.end() && *found == projected);
              binaryEqualChoiceValues_.push_back(
                  static_cast<PnrIndex>(found - values.begin()));
            }
            return offset;
          };
      support.firstChoiceValueOffset = appendChoiceValues(support.first);
      support.secondChoiceValueOffset = appendChoiceValues(support.second);
      support.firstCountOffset = binaryEqualActiveCounts_.size();
      binaryEqualActiveCounts_.resize(binaryEqualActiveCounts_.size() +
                                      support.valueCount);
      support.secondCountOffset = binaryEqualActiveCounts_.size();
      binaryEqualActiveCounts_.resize(binaryEqualActiveCounts_.size() +
                                      support.valueCount);

      const auto appendValueOccurrences = [&](std::size_t choiceValueOffset,
                                              PnrIndex choiceCount) {
        const std::size_t offset = binaryEqualValueOccurrenceOffsets_.size();
        std::vector<std::size_t> counts(support.valueCount, 0);
        for (PnrIndex choice = 0; choice < choiceCount; ++choice)
          ++counts[binaryEqualChoiceValues_[choiceValueOffset + choice]];
        binaryEqualValueOccurrenceOffsets_.push_back(
            binaryEqualChoiceOccurrences_.size());
        for (std::size_t count : counts)
          binaryEqualValueOccurrenceOffsets_.push_back(
              binaryEqualValueOccurrenceOffsets_.back() + count);
        binaryEqualChoiceOccurrences_.resize(
            binaryEqualValueOccurrenceOffsets_.back());
        std::vector<std::size_t> cursors(
            llvm::ArrayRef(binaryEqualValueOccurrenceOffsets_)
                .slice(offset, support.valueCount));
        for (PnrIndex choice = 0; choice < choiceCount; ++choice) {
          const PnrIndex value =
              binaryEqualChoiceValues_[choiceValueOffset + choice];
          binaryEqualChoiceOccurrences_[cursors[value]++] = choice;
        }
        return offset;
      };
      support.firstValueOccurrenceOffset = appendValueOccurrences(
          support.firstChoiceValueOffset,
          decisionChoiceOffsets_[support.first.decision + 1] -
              decisionChoiceOffsets_[support.first.decision]);
      support.secondValueOccurrenceOffset = appendValueOccurrences(
          support.secondChoiceValueOffset,
          decisionChoiceOffsets_[support.second.decision + 1] -
              decisionChoiceOffsets_[support.second.decision]);
      continue;
    }

    if (record.kind != InitializerRelationKind::Disjoint)
      continue;
    std::vector<InitializerRelationMember> orderedMembers(members.begin(),
                                                          members.end());
    llvm::sort(orderedMembers, [](const InitializerRelationMember &lhs,
                                  const InitializerRelationMember &rhs) {
      return lhs.decision < rhs.decision;
    });
    if (llvm::adjacent_find(orderedMembers,
                            [](const InitializerRelationMember &lhs,
                               const InitializerRelationMember &rhs) {
                              return lhs.decision == rhs.decision;
                            }) != orderedMembers.end())
      continue;

    std::vector<PnrIndex> values;
    for (const InitializerRelationMember &member : orderedMembers) {
      const PnrIndex choiceCount = decisionChoiceOffsets_[member.decision + 1] -
                                   decisionChoiceOffsets_[member.decision];
      values.reserve(values.size() + choiceCount);
      for (PnrIndex choice = 0; choice < choiceCount; ++choice)
        values.push_back(model.projectedValue(member, choice));
    }
    llvm::sort(values);
    values.erase(std::unique(values.begin(), values.end()), values.end());
    assert(!values.empty() && values.size() <= getPnrIndexMax());

    AllDifferentRelationSupport &support = allDifferentSupports_[relation];
    support.memberOffset = allDifferentMembers_.size();
    support.memberCount = static_cast<PnrIndex>(orderedMembers.size());
    support.forcedDecisionCountOffset =
        allDifferentForcedDecisionCounts_.size();
    support.valueCount = static_cast<PnrIndex>(values.size());
    rootCardinalityContradiction_ |= support.memberCount > support.valueCount;
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::Seed, [&](llvm::json::Object &fields) {
          fields["operation"] = "initializer_all_different_domain";
          fields["relation"] = relation;
          fields["member_count"] = support.memberCount;
          fields["projected_value_count"] = support.valueCount;
          fields["cardinality_contradiction"] =
              support.memberCount > support.valueCount;
        });
    allDifferentForcedDecisionCounts_.resize(
        allDifferentForcedDecisionCounts_.size() + support.valueCount);
    for (const InitializerRelationMember &member : orderedMembers) {
      AllDifferentMemberSupport memberSupport;
      memberSupport.decision = member.decision;
      memberSupport.choiceValueOffset = allDifferentChoiceValues_.size();
      memberSupport.activeChoiceCountOffset =
          allDifferentActiveChoiceCounts_.size();
      allDifferentActiveChoiceCounts_.resize(
          allDifferentActiveChoiceCounts_.size() + support.valueCount);
      const PnrIndex choiceCount = decisionChoiceOffsets_[member.decision + 1] -
                                   decisionChoiceOffsets_[member.decision];
      for (PnrIndex choice = 0; choice < choiceCount; ++choice) {
        const PnrIndex projected = model.projectedValue(member, choice);
        const auto found = llvm::lower_bound(values, projected);
        assert(found != values.end() && *found == projected);
        allDifferentChoiceValues_.push_back(
            static_cast<PnrIndex>(found - values.begin()));
      }
      allDifferentMembers_.push_back(memberSupport);
    }

    support.valueOccurrenceOffset = allDifferentValueOccurrenceOffsets_.size();
    std::vector<std::size_t> occurrenceCounts(support.valueCount, 0);
    for (std::size_t memberIndex = support.memberOffset;
         memberIndex != support.memberOffset + support.memberCount;
         ++memberIndex) {
      const AllDifferentMemberSupport &member =
          allDifferentMembers_[memberIndex];
      const PnrIndex choiceCount = decisionChoiceOffsets_[member.decision + 1] -
                                   decisionChoiceOffsets_[member.decision];
      for (PnrIndex choice = 0; choice < choiceCount; ++choice)
        ++occurrenceCounts[allDifferentChoiceValues_[member.choiceValueOffset +
                                                     choice]];
    }
    const std::size_t occurrenceBase = allDifferentChoiceOccurrences_.size();
    allDifferentValueOccurrenceOffsets_.push_back(occurrenceBase);
    for (std::size_t count : occurrenceCounts)
      allDifferentValueOccurrenceOffsets_.push_back(
          allDifferentValueOccurrenceOffsets_.back() + count);
    allDifferentChoiceOccurrences_.resize(
        allDifferentValueOccurrenceOffsets_.back());
    std::vector<std::size_t> occurrenceCursors(
        llvm::ArrayRef(allDifferentValueOccurrenceOffsets_)
            .slice(support.valueOccurrenceOffset, support.valueCount));
    for (std::size_t memberIndex = support.memberOffset;
         memberIndex != support.memberOffset + support.memberCount;
         ++memberIndex) {
      const AllDifferentMemberSupport &member =
          allDifferentMembers_[memberIndex];
      const PnrIndex choiceCount = decisionChoiceOffsets_[member.decision + 1] -
                                   decisionChoiceOffsets_[member.decision];
      for (PnrIndex choice = 0; choice < choiceCount; ++choice) {
        const PnrIndex value =
            allDifferentChoiceValues_[member.choiceValueOffset + choice];
        allDifferentChoiceOccurrences_[occurrenceCursors[value]++] = {
            memberIndex, choice};
      }
    }
  }
  allDifferentForcedValuePending_.resize(
      allDifferentForcedDecisionCounts_.size());
  binaryEqualDepletedValuePending_.resize(binaryEqualActiveCounts_.size());
  binaryEqualDepletedValueQueue_.reserve(
      binaryEqualDepletedValuePending_.size());
  allDifferentForcedValueQueue_.reserve(allDifferentMembers_.size());
  PnrIndex maximumAllDifferentMemberCount = 0;
  PnrIndex maximumAllDifferentValueCount = 0;
  for (const AllDifferentRelationSupport &support : allDifferentSupports_) {
    maximumAllDifferentMemberCount =
        std::max(maximumAllDifferentMemberCount, support.memberCount);
    maximumAllDifferentValueCount =
        std::max(maximumAllDifferentValueCount, support.valueCount);
  }
  allDifferentMemberMatches_.resize(maximumAllDifferentMemberCount);
  allDifferentValueMatches_.resize(maximumAllDifferentValueCount);
  allDifferentMemberDistances_.resize(maximumAllDifferentMemberCount);
  allDifferentMemberQueue_.resize(maximumAllDifferentMemberCount);
  allDifferentValueReachable_.resize(maximumAllDifferentValueCount);
  canonicalActiveChoices_.resize(choiceCount);
  choiceOrder_.resize(choiceCount);
  choiceFenwick_.resize(choiceCount);
  completeAssignment_.resize(domainCounts_.size());
  reset();
}

void InitializerRelationSolver::clearQueue() {
  std::fill(relationPending_.begin(), relationPending_.end(), 0);
  std::fill(binaryEqualDepletedValuePending_.begin(),
            binaryEqualDepletedValuePending_.end(), 0);
  binaryEqualDepletedValueQueue_.clear();
  std::fill(allDifferentForcedValuePending_.begin(),
            allDifferentForcedValuePending_.end(), 0);
  allDifferentForcedValueQueue_.clear();
  queueHead_ = 0;
  queueTail_ = 0;
  queueCount_ = 0;
}

void InitializerRelationSolver::reset() {
  std::fill(activeChoices_.begin(), activeChoices_.end(), 1);
  for (PnrIndex decision = 0; decision < domainCounts_.size(); ++decision)
    domainCounts_[decision] =
        decisionChoiceOffsets_[decision + 1] - decisionChoiceOffsets_[decision];
  std::fill(binaryEqualActiveCounts_.begin(), binaryEqualActiveCounts_.end(),
            0);
  std::fill(allDifferentActiveChoiceCounts_.begin(),
            allDifferentActiveChoiceCounts_.end(), 0);
  std::fill(allDifferentForcedDecisionCounts_.begin(),
            allDifferentForcedDecisionCounts_.end(), 0);
  clearQueue();
  for (const BinaryEqualSupport &support : binaryEqualSupports_) {
    if (support.valueCount == 0)
      continue;
    const PnrIndex firstChoiceCount =
        decisionChoiceOffsets_[support.first.decision + 1] -
        decisionChoiceOffsets_[support.first.decision];
    for (PnrIndex choice = 0; choice < firstChoiceCount; ++choice)
      ++binaryEqualActiveCounts_
          [support.firstCountOffset +
           binaryEqualChoiceValues_[support.firstChoiceValueOffset + choice]];
    const PnrIndex secondChoiceCount =
        decisionChoiceOffsets_[support.second.decision + 1] -
        decisionChoiceOffsets_[support.second.decision];
    for (PnrIndex choice = 0; choice < secondChoiceCount; ++choice)
      ++binaryEqualActiveCounts_
          [support.secondCountOffset +
           binaryEqualChoiceValues_[support.secondChoiceValueOffset + choice]];
  }
  for (const AllDifferentRelationSupport &support : allDifferentSupports_) {
    if (support.valueCount == 0)
      continue;
    for (AllDifferentMemberSupport &member :
         llvm::MutableArrayRef(allDifferentMembers_)
             .slice(support.memberOffset, support.memberCount)) {
      member.activeValueCount = 0;
      member.soleActiveValue = getInvalidPnrIndex();
      const PnrIndex choiceCount = decisionChoiceOffsets_[member.decision + 1] -
                                   decisionChoiceOffsets_[member.decision];
      for (PnrIndex choice = 0; choice < choiceCount; ++choice) {
        const PnrIndex value =
            allDifferentChoiceValues_[member.choiceValueOffset + choice];
        PnrIndex &count =
            allDifferentActiveChoiceCounts_[member.activeChoiceCountOffset +
                                            value];
        if (count++ == 0) {
          ++member.activeValueCount;
          member.soleActiveValue = value;
        }
      }
      if (member.activeValueCount == 1)
        ++allDifferentForcedDecisionCounts_[support.forcedDecisionCountOffset +
                                            member.soleActiveValue];
      else
        member.soleActiveValue = getInvalidPnrIndex();
    }
  }
  removalJournal_.clear();
  assignmentAttempts_ = 0;
  propagationInvocationCount_ = 0;
  allDifferentFailureRelation_ = getInvalidPnrIndex();
  allDifferentFailureAtInitialPropagation_ = false;
  for (PnrIndex relation = 0; relation < model_->relations().size();
       ++relation) {
    const BinaryEqualSupport &equal = binaryEqualSupports_[relation];
    if (equal.valueCount != 0) {
      for (PnrIndex value = 0; value < equal.valueCount; ++value) {
        if (binaryEqualActiveCounts_[equal.firstCountOffset + value] == 0)
          enqueueBinaryEqualDepletedValue(relation, value, true);
        if (binaryEqualActiveCounts_[equal.secondCountOffset + value] == 0)
          enqueueBinaryEqualDepletedValue(relation, value, false);
      }
      continue;
    }
    if (allDifferentSupports_[relation].valueCount == 0)
      enqueueRelation(relation);
    else {
      const AllDifferentRelationSupport &support =
          allDifferentSupports_[relation];
      for (PnrIndex value = 0; value < support.valueCount; ++value)
        if (allDifferentForcedDecisionCounts_
                [support.forcedDecisionCountOffset + value] != 0)
          enqueueAllDifferentForcedValue(relation, value);
    }
  }
}

void InitializerRelationSolver::enqueueRelation(PnrIndex relation) {
  if (relationPending_[relation])
    return;
  relationPending_[relation] = 1;
  relationQueue_[queueTail_] = relation;
  queueTail_ =
      relationQueue_.empty() ? 0 : (queueTail_ + 1) % relationQueue_.size();
  ++queueCount_;
}

void InitializerRelationSolver::enqueueDecisionRelations(PnrIndex decision) {
  if (decision >= model_->decisionCount())
    return;
  const auto offsets = model_->decisionRelationOffsets();
  for (PnrIndex relation : model_->decisionRelations().slice(
           offsets[decision], offsets[decision + 1] - offsets[decision])) {
    if (binaryEqualSupports_[relation].valueCount != 0 ||
        allDifferentSupports_[relation].valueCount != 0)
      continue;
    enqueueRelation(relation);
  }
}

void InitializerRelationSolver::updateBinaryEqualSupport(PnrIndex decision,
                                                         PnrIndex localChoice,
                                                         bool add) {
  if (decision >= model_->decisionCount())
    return;
  const auto offsets = model_->decisionRelationOffsets();
  for (PnrIndex relation : model_->decisionRelations().slice(
           offsets[decision], offsets[decision + 1] - offsets[decision])) {
    const BinaryEqualSupport &support = binaryEqualSupports_[relation];
    if (support.valueCount == 0)
      continue;
    std::size_t choiceValueOffset = 0;
    std::size_t countOffset = 0;
    if (support.first.decision == decision) {
      choiceValueOffset = support.firstChoiceValueOffset;
      countOffset = support.firstCountOffset;
    } else {
      assert(support.second.decision == decision);
      choiceValueOffset = support.secondChoiceValueOffset;
      countOffset = support.secondCountOffset;
    }
    PnrIndex &count =
        binaryEqualActiveCounts_[countOffset +
                                 binaryEqualChoiceValues_[choiceValueOffset +
                                                          localChoice]];
    const PnrIndex value =
        binaryEqualChoiceValues_[choiceValueOffset + localChoice];
    if (add) {
      assert(count != getPnrIndexMax());
      ++count;
    } else {
      assert(count != 0);
      if (--count == 0)
        enqueueBinaryEqualDepletedValue(relation, value,
                                        support.first.decision == decision);
    }
  }
}

void InitializerRelationSolver::enqueueBinaryEqualDepletedValue(
    PnrIndex relation, PnrIndex value, bool first) {
  const BinaryEqualSupport &support = binaryEqualSupports_[relation];
  assert(value < support.valueCount);
  const std::size_t countOffset =
      first ? support.firstCountOffset : support.secondCountOffset;
  const std::size_t pendingIndex = countOffset + value;
  if (binaryEqualDepletedValuePending_[pendingIndex])
    return;
  binaryEqualDepletedValuePending_[pendingIndex] = 1;
  binaryEqualDepletedValueQueue_.push_back({relation, value, first});
}

bool InitializerRelationSolver::propagateBinaryEqualDepletedValue(
    PnrIndex relation, PnrIndex value, bool first) {
  const BinaryEqualSupport &support = binaryEqualSupports_[relation];
  const InitializerRelationMember &target =
      first ? support.second : support.first;
  const std::size_t occurrenceOffset = first
                                           ? support.secondValueOccurrenceOffset
                                           : support.firstValueOccurrenceOffset;
  const auto offsets = llvm::ArrayRef(binaryEqualValueOccurrenceOffsets_)
                           .slice(occurrenceOffset, support.valueCount + 1);
  for (PnrIndex localChoice :
       llvm::ArrayRef(binaryEqualChoiceOccurrences_)
           .slice(offsets[value], offsets[value + 1] - offsets[value]))
    if (choiceActive(target.decision, localChoice) &&
        !removeChoice(target.decision, localChoice))
      return false;
  return true;
}

void InitializerRelationSolver::updateAllDifferentSupport(PnrIndex decision,
                                                          PnrIndex localChoice,
                                                          bool add) {
  if (decision >= model_->decisionCount())
    return;
  const auto offsets = model_->decisionRelationOffsets();
  for (PnrIndex relation : model_->decisionRelations().slice(
           offsets[decision], offsets[decision + 1] - offsets[decision])) {
    const AllDifferentRelationSupport &support =
        allDifferentSupports_[relation];
    if (support.valueCount == 0)
      continue;
    auto members = llvm::MutableArrayRef(allDifferentMembers_)
                       .slice(support.memberOffset, support.memberCount);
    auto found = llvm::lower_bound(
        members, decision,
        [](const AllDifferentMemberSupport &member, PnrIndex value) {
          return member.decision < value;
        });
    assert(found != members.end() && found->decision == decision);
    AllDifferentMemberSupport &member = *found;
    const PnrIndex value =
        allDifferentChoiceValues_[member.choiceValueOffset + localChoice];
    PnrIndex &activeChoiceCount =
        allDifferentActiveChoiceCounts_[member.activeChoiceCountOffset + value];
    if (member.activeValueCount == 1) {
      assert(member.soleActiveValue != getInvalidPnrIndex());
      PnrIndex &forced =
          allDifferentForcedDecisionCounts_[support.forcedDecisionCountOffset +
                                            member.soleActiveValue];
      assert(forced != 0);
      --forced;
    }

    if (add) {
      if (activeChoiceCount++ == 0)
        ++member.activeValueCount;
    } else {
      assert(activeChoiceCount != 0 && member.activeValueCount != 0);
      if (--activeChoiceCount == 0)
        --member.activeValueCount;
    }

    member.soleActiveValue = getInvalidPnrIndex();
    if (member.activeValueCount != 1)
      continue;
    for (PnrIndex candidate = 0; candidate < support.valueCount; ++candidate)
      if (allDifferentActiveChoiceCounts_[member.activeChoiceCountOffset +
                                          candidate] != 0) {
        member.soleActiveValue = candidate;
        break;
      }
    assert(member.soleActiveValue != getInvalidPnrIndex());
    ++allDifferentForcedDecisionCounts_[support.forcedDecisionCountOffset +
                                        member.soleActiveValue];
    enqueueAllDifferentForcedValue(relation, member.soleActiveValue);
  }
}

void InitializerRelationSolver::enqueueAllDifferentForcedValue(
    PnrIndex relation, PnrIndex value) {
  const AllDifferentRelationSupport &support = allDifferentSupports_[relation];
  assert(value < support.valueCount);
  const std::size_t pendingIndex = support.forcedDecisionCountOffset + value;
  if (allDifferentForcedValuePending_[pendingIndex])
    return;
  allDifferentForcedValuePending_[pendingIndex] = 1;
  allDifferentForcedValueQueue_.push_back({relation, value});
}

bool InitializerRelationSolver::propagateAllDifferentValue(PnrIndex relation,
                                                           PnrIndex value) {
  const AllDifferentRelationSupport &support = allDifferentSupports_[relation];
  const PnrIndex forcedCount =
      allDifferentForcedDecisionCounts_[support.forcedDecisionCountOffset +
                                        value];
  if (forcedCount == 0)
    return true;
  if (forcedCount != 1)
    return false;

  const auto offsets =
      llvm::ArrayRef(allDifferentValueOccurrenceOffsets_)
          .slice(support.valueOccurrenceOffset, support.valueCount + 1);
  const auto occurrences =
      llvm::ArrayRef(allDifferentChoiceOccurrences_)
          .slice(offsets[value], offsets[value + 1] - offsets[value]);
  std::size_t forcedMember = std::numeric_limits<std::size_t>::max();
  for (const AllDifferentChoiceOccurrence &occurrence : occurrences) {
    const AllDifferentMemberSupport &member =
        allDifferentMembers_[occurrence.member];
    if (member.activeValueCount == 1 && member.soleActiveValue == value) {
      forcedMember = occurrence.member;
      break;
    }
  }
  assert(forcedMember != std::numeric_limits<std::size_t>::max());
  for (const AllDifferentChoiceOccurrence &occurrence : occurrences) {
    if (occurrence.member == forcedMember)
      continue;
    const PnrIndex decision = allDifferentMembers_[occurrence.member].decision;
    if (choiceActive(decision, occurrence.localChoice) &&
        !removeChoice(decision, occurrence.localChoice))
      return false;
  }
  return true;
}

bool InitializerRelationSolver::augmentAllDifferentMatching(
    const AllDifferentRelationSupport &support, PnrIndex memberOrdinal,
    PnrIndex shortestLength) {
  const auto members = llvm::ArrayRef(allDifferentMembers_)
                           .slice(support.memberOffset, support.memberCount);
  const AllDifferentMemberSupport &member = members[memberOrdinal];
  const PnrIndex nextDistance = allDifferentMemberDistances_[memberOrdinal] + 1;
  for (PnrIndex value = 0; value < support.valueCount; ++value) {
    if (allDifferentActiveChoiceCounts_[member.activeChoiceCountOffset +
                                        value] == 0)
      continue;
    const PnrIndex owner = allDifferentValueMatches_[value];
    if (owner == getInvalidPnrIndex()) {
      if (nextDistance != shortestLength)
        continue;
    } else if (allDifferentMemberDistances_[owner] != nextDistance ||
               !augmentAllDifferentMatching(support, owner, shortestLength)) {
      continue;
    }
    allDifferentMemberMatches_[memberOrdinal] = value;
    allDifferentValueMatches_[value] = memberOrdinal;
    return true;
  }
  allDifferentMemberDistances_[memberOrdinal] = getInvalidPnrIndex();
  return false;
}

bool InitializerRelationSolver::allDifferentMatchingFeasible(
    PnrIndex relation, const AllDifferentRelationSupport &support,
    bool initialPropagation) {
  if (support.valueCount == 0)
    return true;
  auto memberMatches = llvm::MutableArrayRef(allDifferentMemberMatches_)
                           .take_front(support.memberCount);
  auto valueMatches = llvm::MutableArrayRef(allDifferentValueMatches_)
                          .take_front(support.valueCount);
  std::fill(memberMatches.begin(), memberMatches.end(), getInvalidPnrIndex());
  std::fill(valueMatches.begin(), valueMatches.end(), getInvalidPnrIndex());

  PnrIndex matched = 0;
  while (matched != support.memberCount) {
    auto distances = llvm::MutableArrayRef(allDifferentMemberDistances_)
                         .take_front(support.memberCount);
    std::fill(distances.begin(), distances.end(), getInvalidPnrIndex());
    PnrIndex head = 0;
    PnrIndex tail = 0;
    for (PnrIndex member = 0; member < support.memberCount; ++member) {
      if (memberMatches[member] != getInvalidPnrIndex())
        continue;
      distances[member] = 0;
      allDifferentMemberQueue_[tail++] = member;
    }

    PnrIndex shortestLength = getInvalidPnrIndex();
    const auto members = llvm::ArrayRef(allDifferentMembers_)
                             .slice(support.memberOffset, support.memberCount);
    while (head != tail) {
      const PnrIndex memberOrdinal = allDifferentMemberQueue_[head++];
      const PnrIndex nextDistance = distances[memberOrdinal] + 1;
      if (shortestLength != getInvalidPnrIndex() &&
          nextDistance > shortestLength)
        continue;
      const AllDifferentMemberSupport &member = members[memberOrdinal];
      for (PnrIndex value = 0; value < support.valueCount; ++value) {
        if (allDifferentActiveChoiceCounts_[member.activeChoiceCountOffset +
                                            value] == 0)
          continue;
        const PnrIndex owner = valueMatches[value];
        if (owner == getInvalidPnrIndex()) {
          shortestLength = nextDistance;
          continue;
        }
        if (distances[owner] != getInvalidPnrIndex())
          continue;
        distances[owner] = nextDistance;
        allDifferentMemberQueue_[tail++] = owner;
      }
    }
    if (shortestLength == getInvalidPnrIndex()) {
      auto valueReachable = llvm::MutableArrayRef(allDifferentValueReachable_)
                                .take_front(support.valueCount);
      std::fill(valueReachable.begin(), valueReachable.end(), 0);
      PnrIndex hallMemberCount = 0;
      for (PnrIndex memberOrdinal = 0; memberOrdinal < support.memberCount;
           ++memberOrdinal) {
        if (distances[memberOrdinal] == getInvalidPnrIndex())
          continue;
        ++hallMemberCount;
        const AllDifferentMemberSupport &member = members[memberOrdinal];
        for (PnrIndex value = 0; value < support.valueCount; ++value)
          if (allDifferentActiveChoiceCounts_[member.activeChoiceCountOffset +
                                              value] != 0)
            valueReachable[value] = 1;
      }
      const PnrIndex hallValueCount = static_cast<PnrIndex>(
          llvm::count(valueReachable, static_cast<std::uint8_t>(1)));
      allDifferentFailureRelation_ = relation;
      allDifferentFailureMatched_ = matched;
      allDifferentFailureMemberCount_ = hallMemberCount;
      allDifferentFailureValueCount_ = hallValueCount;
      allDifferentFailureAtInitialPropagation_ = initialPropagation;
      if (!initialPropagation)
        return false;
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Summary,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::Seed, [&](llvm::json::Object &fields) {
            fields["operation"] = "initializer_all_different_hall_failure";
            fields["relation"] = relation;
            fields["member_count"] = support.memberCount;
            fields["projected_value_count"] = support.valueCount;
            fields["maximum_matching_size"] = matched;
            fields["hall_member_count"] = hallMemberCount;
            fields["hall_value_count"] = hallValueCount;
          });
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::Seed, [&](llvm::json::Object &fields) {
            fields["operation"] = "initializer_all_different_hall_witness";
            fields["relation"] = relation;
            llvm::json::Array decisions;
            for (PnrIndex memberOrdinal = 0;
                 memberOrdinal < support.memberCount; ++memberOrdinal)
              if (distances[memberOrdinal] != getInvalidPnrIndex())
                decisions.emplace_back(members[memberOrdinal].decision);
            llvm::json::Array values;
            for (PnrIndex value = 0; value < support.valueCount; ++value)
              if (valueReachable[value] != 0)
                values.emplace_back(value);
            fields["hall_decisions"] = std::move(decisions);
            fields["hall_values"] = std::move(values);
          });
      return false;
    }

    PnrIndex augmented = 0;
    for (PnrIndex member = 0; member < support.memberCount; ++member)
      if (memberMatches[member] == getInvalidPnrIndex() &&
          augmentAllDifferentMatching(support, member, shortestLength))
        ++augmented;
    if (augmented == 0)
      return false;
    matched += augmented;
  }
  return true;
}

bool InitializerRelationSolver::choiceActive(PnrIndex decision,
                                             PnrIndex localChoice) const {
  return activeChoices_[decisionChoiceOffsets_[decision] + localChoice] != 0;
}

bool InitializerRelationSolver::removeChoice(PnrIndex decision,
                                             PnrIndex localChoice) {
  const PnrIndex choice = decisionChoiceOffsets_[decision] + localChoice;
  if (!activeChoices_[choice])
    return domainCounts_[decision] != 0;
  activeChoices_[choice] = 0;
  updateBinaryEqualSupport(decision, localChoice, false);
  updateAllDifferentSupport(decision, localChoice, false);
  --domainCounts_[decision];
  removalJournal_.push_back({decision, localChoice});
  enqueueDecisionRelations(decision);
  return domainCounts_[decision] != 0;
}

bool InitializerRelationSolver::equalChoiceSupported(
    PnrIndex relationOrdinal, PnrIndex decision, PnrIndex localChoice) const {
  const BinaryEqualSupport &support = binaryEqualSupports_[relationOrdinal];
  if (support.valueCount != 0) {
    std::size_t choiceValueOffset = 0;
    std::size_t otherCountOffset = 0;
    if (support.first.decision == decision) {
      choiceValueOffset = support.firstChoiceValueOffset;
      otherCountOffset = support.secondCountOffset;
    } else {
      assert(support.second.decision == decision);
      choiceValueOffset = support.secondChoiceValueOffset;
      otherCountOffset = support.firstCountOffset;
    }
    const PnrIndex value =
        binaryEqualChoiceValues_[choiceValueOffset + localChoice];
    return binaryEqualActiveCounts_[otherCountOffset + value] != 0;
  }

  const InitializerRelationRecord &relation =
      model_->relations()[relationOrdinal];
  const auto members = model_->members(relation);
  std::optional<PnrIndex> required;
  for (const InitializerRelationMember &member : members) {
    if (member.decision != decision)
      continue;
    const PnrIndex value = model_->projectedValue(member, localChoice);
    if (required && *required != value)
      return false;
    required = value;
  }
  if (!required)
    return false;

  for (std::size_t memberOrdinal = 0; memberOrdinal < members.size();
       ++memberOrdinal) {
    const PnrIndex other = members[memberOrdinal].decision;
    if (other == decision)
      continue;
    bool firstOccurrence = true;
    for (std::size_t earlier = 0; earlier < memberOrdinal; ++earlier)
      firstOccurrence &= members[earlier].decision != other;
    if (!firstOccurrence)
      continue;

    bool supported = false;
    const PnrIndex count =
        domainCounts_[other] == 0
            ? 0
            : decisionChoiceOffsets_[other + 1] - decisionChoiceOffsets_[other];
    for (PnrIndex choice = 0; choice < count && !supported; ++choice) {
      if (!choiceActive(other, choice))
        continue;
      supported = true;
      for (const InitializerRelationMember &otherMember : members)
        if (otherMember.decision == other &&
            model_->projectedValue(otherMember, choice) != *required) {
          supported = false;
          break;
        }
    }
    if (!supported)
      return false;
  }
  return true;
}

bool InitializerRelationSolver::disjointChoiceSupported(
    PnrIndex relationOrdinal, PnrIndex decision, PnrIndex localChoice) const {
  const AllDifferentRelationSupport &support =
      allDifferentSupports_[relationOrdinal];
  if (support.valueCount != 0) {
    const auto members = llvm::ArrayRef(allDifferentMembers_)
                             .slice(support.memberOffset, support.memberCount);
    const auto found = llvm::lower_bound(
        members, decision,
        [](const AllDifferentMemberSupport &member, PnrIndex value) {
          return member.decision < value;
        });
    assert(found != members.end() && found->decision == decision);
    const PnrIndex value =
        allDifferentChoiceValues_[found->choiceValueOffset + localChoice];
    const PnrIndex forced =
        allDifferentForcedDecisionCounts_[support.forcedDecisionCountOffset +
                                          value];
    const bool selfForced =
        found->activeValueCount == 1 && found->soleActiveValue == value;
    return forced <= static_cast<PnrIndex>(selfForced);
  }

  const InitializerRelationRecord &relation =
      model_->relations()[relationOrdinal];
  const auto members = model_->members(relation);
  const auto collidesWithDecision = [&](PnrIndex value) {
    for (const InitializerRelationMember &member : members)
      if (member.decision == decision &&
          model_->projectedValue(member, localChoice) == value)
        return true;
    return false;
  };
  for (std::size_t lhs = 0; lhs < members.size(); ++lhs) {
    if (members[lhs].decision != decision)
      continue;
    for (std::size_t rhs = lhs + 1; rhs < members.size(); ++rhs)
      if (members[rhs].decision == decision &&
          model_->projectedValue(members[lhs], localChoice) ==
              model_->projectedValue(members[rhs], localChoice))
        return false;
  }

  for (std::size_t memberOrdinal = 0; memberOrdinal < members.size();
       ++memberOrdinal) {
    const PnrIndex other = members[memberOrdinal].decision;
    if (other == decision)
      continue;
    bool firstOccurrence = true;
    for (std::size_t earlier = 0; earlier < memberOrdinal; ++earlier)
      firstOccurrence &= members[earlier].decision != other;
    if (!firstOccurrence)
      continue;

    bool supported = false;
    const PnrIndex count =
        decisionChoiceOffsets_[other + 1] - decisionChoiceOffsets_[other];
    for (PnrIndex choice = 0; choice < count && !supported; ++choice) {
      if (!choiceActive(other, choice))
        continue;
      supported = true;
      for (std::size_t lhs = 0; lhs < members.size() && supported; ++lhs) {
        if (members[lhs].decision != other)
          continue;
        const PnrIndex value = model_->projectedValue(members[lhs], choice);
        if (collidesWithDecision(value)) {
          supported = false;
          break;
        }
        for (std::size_t rhs = lhs + 1; rhs < members.size(); ++rhs)
          if (members[rhs].decision == other &&
              model_->projectedValue(members[rhs], choice) == value) {
            supported = false;
            break;
          }
      }
    }
    if (!supported)
      return false;
  }
  return true;
}

bool InitializerRelationSolver::capacityChoiceSupported(
    PnrIndex relationOrdinal, PnrIndex decision, PnrIndex localChoice) const {
  const InitializerRelationRecord &relation =
      model_->relations()[relationOrdinal];
  const auto capacities = model_->valueCapacities(relation);
  std::vector<std::uint64_t> forcedLoads(capacities.size(), 0);
  for (const InitializerRelationMember &member : model_->members(relation)) {
    std::optional<PnrIndex> forcedValue;
    if (member.decision == decision) {
      forcedValue = model_->projectedValue(member, localChoice);
    } else {
      const PnrIndex count = decisionChoiceOffsets_[member.decision + 1] -
                             decisionChoiceOffsets_[member.decision];
      bool varies = false;
      for (PnrIndex choice = 0; choice < count; ++choice) {
        if (!choiceActive(member.decision, choice))
          continue;
        const PnrIndex value = model_->projectedValue(member, choice);
        if (!forcedValue)
          forcedValue = value;
        else if (*forcedValue != value) {
          varies = true;
          break;
        }
      }
      if (varies)
        continue;
    }
    assert(forcedValue && *forcedValue < capacities.size());
    if (member.demand > capacities[*forcedValue] - forcedLoads[*forcedValue])
      return false;
    forcedLoads[*forcedValue] += member.demand;
  }
  return true;
}

bool InitializerRelationSolver::relationChoiceSupported(
    PnrIndex relation, PnrIndex decision, PnrIndex localChoice) const {
  const InitializerRelationRecord &record = model_->relations()[relation];
  switch (record.kind) {
  case InitializerRelationKind::Equal:
    return equalChoiceSupported(relation, decision, localChoice);
  case InitializerRelationKind::Disjoint:
    return disjointChoiceSupported(relation, decision, localChoice);
  case InitializerRelationKind::Capacity:
    return capacityChoiceSupported(relation, decision, localChoice);
  }
  llvm_unreachable("unknown initializer relation kind");
}

bool InitializerRelationSolver::activeRelationSatisfied(
    const InitializerRelationRecord &relation) const {
  const auto members = model_->members(relation);
  if (relation.kind == InitializerRelationKind::Capacity) {
    const auto capacities = model_->valueCapacities(relation);
    std::vector<std::uint64_t> loads(capacities.size(), 0);
    for (const InitializerRelationMember &member : members) {
      const PnrIndex value =
          model_->projectedValue(member, soleChoice(member.decision));
      assert(value < capacities.size());
      if (member.demand > capacities[value] - loads[value])
        return false;
      loads[value] += member.demand;
    }
    return true;
  }
  std::optional<PnrIndex> equalValue;
  for (std::size_t lhs = 0; lhs < members.size(); ++lhs) {
    const InitializerRelationMember &member = members[lhs];
    const PnrIndex value =
        model_->projectedValue(member, soleChoice(member.decision));
    if (relation.kind == InitializerRelationKind::Equal) {
      if (equalValue && *equalValue != value)
        return false;
      equalValue = value;
      continue;
    }
    for (std::size_t rhs = 0; rhs < lhs; ++rhs) {
      const InitializerRelationMember &other = members[rhs];
      if (model_->projectedValue(other, soleChoice(other.decision)) == value)
        return false;
    }
  }
  return true;
}

bool InitializerRelationSolver::propagate() {
  const bool initialPropagation = propagationInvocationCount_++ == 0;
  allDifferentFailureRelation_ = getInvalidPnrIndex();
  allDifferentFailureAtInitialPropagation_ = false;
  while (queueCount_ != 0 || !allDifferentForcedValueQueue_.empty() ||
         !binaryEqualDepletedValueQueue_.empty()) {
    if (!allDifferentForcedValueQueue_.empty()) {
      const AllDifferentForcedValue forced =
          allDifferentForcedValueQueue_.back();
      allDifferentForcedValueQueue_.pop_back();
      const AllDifferentRelationSupport &support =
          allDifferentSupports_[forced.relation];
      allDifferentForcedValuePending_[support.forcedDecisionCountOffset +
                                      forced.value] = 0;
      if (!propagateAllDifferentValue(forced.relation, forced.value))
        return false;
      continue;
    }
    if (!binaryEqualDepletedValueQueue_.empty()) {
      const BinaryEqualDepletedValue depleted =
          binaryEqualDepletedValueQueue_.back();
      binaryEqualDepletedValueQueue_.pop_back();
      const BinaryEqualSupport &support =
          binaryEqualSupports_[depleted.relation];
      const std::size_t countOffset =
          depleted.first ? support.firstCountOffset : support.secondCountOffset;
      binaryEqualDepletedValuePending_[countOffset + depleted.value] = 0;
      if (!propagateBinaryEqualDepletedValue(depleted.relation, depleted.value,
                                             depleted.first))
        return false;
      continue;
    }
    const PnrIndex relation = relationQueue_[queueHead_];
    queueHead_ = (queueHead_ + 1) % relationQueue_.size();
    --queueCount_;
    relationPending_[relation] = 0;
    const InitializerRelationRecord &record = model_->relations()[relation];
    const auto members = model_->members(record);
    for (std::size_t memberOrdinal = 0; memberOrdinal < members.size();
         ++memberOrdinal) {
      const PnrIndex decision = members[memberOrdinal].decision;
      bool firstOccurrence = true;
      for (std::size_t earlier = 0; earlier < memberOrdinal; ++earlier)
        firstOccurrence &= members[earlier].decision != decision;
      if (!firstOccurrence)
        continue;
      const PnrIndex count = decisionChoiceOffsets_[decision + 1] -
                             decisionChoiceOffsets_[decision];
      for (PnrIndex choice = 0; choice < count; ++choice)
        if (choiceActive(decision, choice) &&
            !relationChoiceSupported(relation, decision, choice) &&
            !removeChoice(decision, choice))
          return false;
    }
  }
  for (auto [relation, support] : llvm::enumerate(allDifferentSupports_))
    if (!allDifferentMatchingFeasible(static_cast<PnrIndex>(relation), support,
                                      initialPropagation))
      return false;
  return true;
}

std::string InitializerRelationSolver::allDifferentHallFailureMessage() const {
  assert(allDifferentFailureRelation_ < allDifferentSupports_.size());
  return "initializer all-different relation " +
         std::to_string(allDifferentFailureRelation_) +
         " has a Hall deficit: maximum matching " +
         std::to_string(allDifferentFailureMatched_) + "/" +
         std::to_string(
             allDifferentSupports_[allDifferentFailureRelation_].memberCount) +
         ", witness members " +
         std::to_string(allDifferentFailureMemberCount_) + ", witness values " +
         std::to_string(allDifferentFailureValueCount_);
}

PnrIndex InitializerRelationSolver::soleChoice(PnrIndex decision) const {
  assert(domainCounts_[decision] == 1);
  const PnrIndex count =
      decisionChoiceOffsets_[decision + 1] - decisionChoiceOffsets_[decision];
  for (PnrIndex choice = 0; choice < count; ++choice)
    if (choiceActive(decision, choice))
      return choice;
  llvm_unreachable("singleton initializer domain has no active choice");
}

void InitializerRelationSolver::rollback(std::size_t journalMark) {
  clearQueue();
  while (removalJournal_.size() > journalMark) {
    const RemovedChoice removed = removalJournal_.back();
    removalJournal_.pop_back();
    const PnrIndex choice =
        decisionChoiceOffsets_[removed.decision] + removed.localChoice;
    assert(!activeChoices_[choice]);
    activeChoices_[choice] = 1;
    updateBinaryEqualSupport(removed.decision, removed.localChoice, true);
    updateAllDifferentSupport(removed.decision, removed.localChoice, true);
    ++domainCounts_[removed.decision];
  }
}

llvm::Expected<InitializerRelationSolver::SearchResult>
InitializerRelationSolver::search(
    std::uint64_t assignmentLimit,
    DeterministicPnrRandomStream *diversificationStream,
    llvm::ArrayRef<PnrIndex> preferredChoices,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
        validateCompleteAssignment) {
  if (!propagate())
    return SearchResult::Contradiction;

  PnrIndex selected = getInvalidPnrIndex();
  PnrIndex selectedCount = getInvalidPnrIndex();
  for (PnrIndex decision = 0; decision < domainCounts_.size(); ++decision)
    if (domainCounts_[decision] > 1 &&
        domainCounts_[decision] < selectedCount) {
      selected = decision;
      selectedCount = domainCounts_[decision];
    }
  if (selected == getInvalidPnrIndex()) {
    for (const InitializerRelationRecord &relation : model_->relations())
      if (!activeRelationSatisfied(relation))
        return SearchResult::Contradiction;
    for (PnrIndex decision = 0; decision < domainCounts_.size(); ++decision)
      completeAssignment_[decision] = soleChoice(decision);
    auto accepted = validateCompleteAssignment(completeAssignment_);
    if (!accepted)
      return accepted.takeError();
    return *accepted ? SearchResult::Solved : SearchResult::Contradiction;
  }

  const PnrIndex choiceCount =
      decisionChoiceOffsets_[selected + 1] - decisionChoiceOffsets_[selected];
  const llvm::ArrayRef<PnrIndex> choiceOrder =
      buildChoiceOrder(selected, diversificationStream, preferredChoices);
  for (PnrIndex selectedChoice : choiceOrder) {
    if (assignmentAttempts_ == assignmentLimit)
      return SearchResult::WorkLimit;
    ++assignmentAttempts_;
    const std::size_t journalMark = removalJournal_.size();
    bool retainedChoice = true;
    for (PnrIndex choice = 0; choice < choiceCount; ++choice)
      if (choice != selectedChoice && choiceActive(selected, choice))
        retainedChoice &= removeChoice(selected, choice);
    auto result =
        retainedChoice
            ? search(assignmentLimit, diversificationStream, preferredChoices,
                     validateCompleteAssignment)
            : llvm::Expected<SearchResult>(SearchResult::Contradiction);
    if (!result)
      return result.takeError();
    if (*result != SearchResult::Contradiction)
      return *result;
    rollback(journalMark);
  }
  return SearchResult::Contradiction;
}

llvm::ArrayRef<PnrIndex> InitializerRelationSolver::buildChoiceOrder(
    PnrIndex decision, DeterministicPnrRandomStream *diversificationStream,
    llvm::ArrayRef<PnrIndex> preferredChoices) {
  const PnrIndex choiceOffset = decisionChoiceOffsets_[decision];
  const PnrIndex choiceCount =
      decisionChoiceOffsets_[decision + 1] - choiceOffset;
  PnrIndex activeCount = 0;
  for (PnrIndex choice = 0; choice < choiceCount; ++choice)
    if (choiceActive(decision, choice))
      canonicalActiveChoices_[choiceOffset + activeCount++] = choice;
  assert(activeCount == domainCounts_[decision]);

  llvm::cantFail(buildInitializerChoiceOrder(
      llvm::ArrayRef(canonicalActiveChoices_).slice(choiceOffset, activeCount),
      diversificationStream,
      llvm::MutableArrayRef(choiceOrder_).slice(choiceOffset, activeCount),
      llvm::MutableArrayRef(choiceFenwick_).slice(choiceOffset, activeCount)));
  if (!diversificationStream && !preferredChoices.empty()) {
    const PnrIndex preferred = preferredChoices[decision];
    auto order =
        llvm::MutableArrayRef(choiceOrder_).slice(choiceOffset, activeCount);
    const auto found = llvm::find(order, preferred);
    if (preferred != getInvalidPnrIndex() && found != order.end())
      std::rotate(order.begin(), found, std::next(found));
  }
  return llvm::ArrayRef(choiceOrder_).slice(choiceOffset, activeCount);
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonical(std::uint64_t assignmentLimit) {
  return solve(
      assignmentLimit, nullptr, {},
      [](llvm::ArrayRef<PnrIndex>) -> llvm::Expected<bool> { return true; });
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonical(
    std::uint64_t assignmentLimit,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
        validateCompleteAssignment) {
  return solve(assignmentLimit, nullptr, {}, validateCompleteAssignment);
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonicalWithFixedChoices(
    std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices) {
  return solveCanonicalWithFixedChoices(
      assignmentLimit, fixedChoices,
      [](llvm::ArrayRef<PnrIndex>) -> llvm::Expected<bool> { return true; });
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonicalWithReleasedChoices(
    std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
    llvm::ArrayRef<PnrIndex> releasedDecisions) {
  assignmentAttempts_ = 0;
  if (fixedChoices.size() != domainCounts_.size())
    return assignmentError(
        "fixed choice count does not match the decision domain");

  std::vector<PnrIndex> released(releasedDecisions.begin(),
                                 releasedDecisions.end());
  llvm::sort(released);
  released.erase(std::unique(released.begin(), released.end()), released.end());
  std::vector<std::uint8_t> releasedMarks(domainCounts_.size(), 0);
  for (PnrIndex decision : released) {
    if (decision >= domainCounts_.size())
      return assignmentError("a released decision is outside the domain");
    releasedMarks[decision] = 1;
  }

  for (PnrIndex decision = 0; decision < fixedChoices.size(); ++decision) {
    const PnrIndex fixed = fixedChoices[decision];
    if (fixed == getInvalidPnrIndex()) {
      if (!releasedMarks[decision])
        return assignmentError("a retained decision has no fixed choice");
      continue;
    }
    const PnrIndex choiceCount =
        decisionChoiceOffsets_[decision + 1] - decisionChoiceOffsets_[decision];
    if (fixed >= choiceCount)
      return assignmentError("a fixed choice is outside its decision domain");
  }

  std::vector<PnrIndex> reducedDecision(releasedMarks.size(),
                                        getInvalidPnrIndex());
  std::vector<PnrIndex> reducedChoiceCounts;
  reducedChoiceCounts.reserve(released.size());
  for (PnrIndex decision : released) {
    reducedDecision[decision] =
        static_cast<PnrIndex>(reducedChoiceCounts.size());
    reducedChoiceCounts.push_back(decisionChoiceOffsets_[decision + 1] -
                                  decisionChoiceOffsets_[decision]);
  }

  const auto appendReleasedMember = [&](const InitializerRelationMember &member,
                                        InitializerRelationInput &relation) {
    std::vector<PnrIndex> projectedValues;
    const PnrIndex choiceCount = decisionChoiceOffsets_[member.decision + 1] -
                                 decisionChoiceOffsets_[member.decision];
    projectedValues.reserve(choiceCount);
    for (PnrIndex choice = 0; choice < choiceCount; ++choice)
      projectedValues.push_back(model_->projectedValue(member, choice));
    relation.members.push_back({reducedDecision[member.decision],
                                std::move(projectedValues), member.demand});
  };
  const auto appendFixedValue =
      [&](PnrIndex value, InitializerRelationInput &relation) -> llvm::Error {
    if (reducedChoiceCounts.size() == getPnrIndexMax())
      return modelError("projected decision count overflows PnrIndex");
    const PnrIndex decision = static_cast<PnrIndex>(reducedChoiceCounts.size());
    reducedChoiceCounts.push_back(1);
    relation.members.push_back({decision, {value}});
    return llvm::Error::success();
  };
  const auto fixedInfeasible = [](const llvm::Twine &message) -> llvm::Error {
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::FixedRootInfeasible,
        message.str());
  };

  std::vector<InitializerRelationInput> reducedRelations;
  reducedRelations.reserve(model_->relations().size());
  const auto modelFixedChoices =
      fixedChoices.take_front(model_->decisionCount());
  for (PnrIndex relationOrdinal = 0;
       relationOrdinal < model_->relations().size(); ++relationOrdinal) {
    const InitializerRelationRecord &record =
        model_->relations()[relationOrdinal];
    const auto members = model_->members(record);
    const bool hasReleasedMember =
        llvm::any_of(members, [&](const InitializerRelationMember &member) {
          return releasedMarks[member.decision] != 0;
        });
    if (!hasReleasedMember) {
      if (!model_->relationSatisfied(relationOrdinal, modelFixedChoices))
        return fixedInfeasible(
            "retained initializer choices violate a hard relation");
      continue;
    }

    InitializerRelationInput reduced;
    reduced.kind = record.kind;
    if (record.kind == InitializerRelationKind::Capacity) {
      const auto capacities = model_->valueCapacities(record);
      reduced.valueCapacities.assign(capacities.begin(), capacities.end());
      for (const InitializerRelationMember &member : members) {
        if (releasedMarks[member.decision]) {
          appendReleasedMember(member, reduced);
          continue;
        }
        const PnrIndex value =
            model_->projectedValue(member, fixedChoices[member.decision]);
        if (member.demand > reduced.valueCapacities[value])
          return fixedInfeasible(
              "retained initializer choices exceed relation capacity");
        reduced.valueCapacities[value] -= member.demand;
      }
      reducedRelations.push_back(std::move(reduced));
      continue;
    }

    std::optional<PnrIndex> equalFixedValue;
    std::vector<PnrIndex> disjointFixedValues;
    if (record.kind == InitializerRelationKind::Disjoint)
      disjointFixedValues.reserve(members.size());
    for (const InitializerRelationMember &member : members) {
      if (releasedMarks[member.decision]) {
        appendReleasedMember(member, reduced);
        continue;
      }
      const PnrIndex value =
          model_->projectedValue(member, fixedChoices[member.decision]);
      if (record.kind == InitializerRelationKind::Equal) {
        if (equalFixedValue && *equalFixedValue != value)
          return fixedInfeasible(
              "retained initializer choices violate equality");
        equalFixedValue = value;
        continue;
      }
      if (llvm::is_contained(disjointFixedValues, value))
        return fixedInfeasible(
            "retained initializer choices violate disjointness");
      disjointFixedValues.push_back(value);
    }
    if (equalFixedValue) {
      if (llvm::Error error = appendFixedValue(*equalFixedValue, reduced))
        return std::move(error);
    } else {
      for (PnrIndex value : disjointFixedValues)
        if (llvm::Error error = appendFixedValue(value, reduced))
          return std::move(error);
    }
    reducedRelations.push_back(std::move(reduced));
  }

  auto reducedModel = InitializerRelationModel::create(
      std::move(reducedChoiceCounts), std::move(reducedRelations));
  if (!reducedModel)
    return reducedModel.takeError();
  InitializerRelationSolver reducedSolver(*reducedModel);
  auto reducedResult = reducedSolver.solveCanonical(assignmentLimit);
  assignmentAttempts_ = reducedSolver.assignmentAttempts();
  if (!reducedResult) {
    llvm::Error translated = llvm::handleErrors(
        reducedResult.takeError(),
        [&](const InitializerRelationSolveFailure &failure) -> llvm::Error {
          std::string message;
          llvm::raw_string_ostream stream(message);
          failure.log(stream);
          return llvm::make_error<InitializerRelationSolveFailure>(
              failure.kind() == InitializerRelationSolveFailureKind::WorkLimit
                  ? InitializerRelationSolveFailureKind::WorkLimit
                  : InitializerRelationSolveFailureKind::FixedRootInfeasible,
              stream.str());
        });
    return std::move(translated);
  }

  InitializerRelationSolveResult solved;
  solved.choices.assign(fixedChoices.begin(), fixedChoices.end());
  for (auto [local, decision] : llvm::enumerate(released))
    solved.choices[decision] = reducedResult->choices[local];
  if (llvm::Error error = model_->verifyChoices(
          llvm::ArrayRef(solved.choices).take_front(model_->decisionCount())))
    return std::move(error);
  solved.assignmentAttempts = reducedResult->assignmentAttempts;
  return solved;
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonicalWithFixedChoices(
    std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
        validateCompleteAssignment) {
  return solveCanonicalWithFixedAndPreferredChoices(
      assignmentLimit, fixedChoices, {}, validateCompleteAssignment);
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonicalWithPreferredChoices(
    std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> preferredChoices) {
  std::vector<PnrIndex> fixed(domainCounts_.size(), getInvalidPnrIndex());
  return solveCanonicalWithFixedAndPreferredChoices(
      assignmentLimit, fixed, preferredChoices,
      [](llvm::ArrayRef<PnrIndex>) -> llvm::Expected<bool> { return true; });
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonicalWithFixedAndPreferredChoices(
    std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
    llvm::ArrayRef<PnrIndex> preferredChoices) {
  return solveCanonicalWithFixedAndPreferredChoices(
      assignmentLimit, fixedChoices, preferredChoices,
      [](llvm::ArrayRef<PnrIndex>) -> llvm::Expected<bool> { return true; });
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonicalWithFixedAndPreferredChoices(
    std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
    llvm::ArrayRef<PnrIndex> preferredChoices,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
        validateCompleteAssignment) {
  if (fixedChoices.size() != domainCounts_.size())
    return assignmentError(
        "fixed choice count does not match the decision domain");
  if (!preferredChoices.empty() &&
      preferredChoices.size() != domainCounts_.size())
    return assignmentError(
        "preferred choice count does not match the decision domain");
  for (PnrIndex decision = 0; decision < preferredChoices.size(); ++decision) {
    const PnrIndex preferred = preferredChoices[decision];
    if (preferred != getInvalidPnrIndex() &&
        preferred >= decisionChoiceOffsets_[decision + 1] -
                         decisionChoiceOffsets_[decision])
      return assignmentError(
          "a preferred choice is outside its decision domain");
  }
  if (rootCardinalityContradiction_) {
    assignmentAttempts_ = 0;
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::ProvenInfeasible,
        "initializer all-different relation has fewer values than members");
  }
  reset();
  for (PnrIndex decision = 0; decision < fixedChoices.size(); ++decision) {
    const PnrIndex fixed = fixedChoices[decision];
    if (fixed == getInvalidPnrIndex())
      continue;
    const PnrIndex choiceCount =
        decisionChoiceOffsets_[decision + 1] - decisionChoiceOffsets_[decision];
    if (fixed >= choiceCount)
      return assignmentError("a fixed choice is outside its decision domain");
    for (PnrIndex choice = 0; choice < choiceCount; ++choice)
      if (choice != fixed && !removeChoice(decision, choice))
        return llvm::make_error<InitializerRelationSolveFailure>(
            InitializerRelationSolveFailureKind::FixedRootInfeasible,
            "Spatial initializer fixed choices are infeasible");
  }

  auto result = search(assignmentLimit, nullptr, preferredChoices,
                       validateCompleteAssignment);
  if (!result)
    return result.takeError();
  if (*result == SearchResult::WorkLimit)
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::WorkLimit,
        "Spatial initializer exhausted its assignment work limit");
  if (*result == SearchResult::Contradiction) {
    if (allDifferentFailureAtInitialPropagation_)
      return llvm::make_error<InitializerRelationSolveFailure>(
          InitializerRelationSolveFailureKind::FixedRootInfeasible,
          allDifferentHallFailureMessage());
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::FixedRootInfeasible,
        "Spatial initializer fixed choices are infeasible");
  }

  InitializerRelationSolveResult solved;
  solved.choices.reserve(domainCounts_.size());
  for (PnrIndex decision = 0; decision < domainCounts_.size(); ++decision)
    solved.choices.push_back(soleChoice(decision));
  solved.assignmentAttempts = assignmentAttempts_;
  return solved;
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveDiversified(
    std::uint64_t assignmentLimit,
    DeterministicPnrRandomStream &diversificationStream) {
  return solveDiversified(
      assignmentLimit, diversificationStream,
      [](llvm::ArrayRef<PnrIndex>) -> llvm::Expected<bool> { return true; });
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveDiversified(
    std::uint64_t assignmentLimit,
    DeterministicPnrRandomStream &diversificationStream,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
        validateCompleteAssignment) {
  return solve(assignmentLimit, &diversificationStream, {},
               validateCompleteAssignment);
}

llvm::Expected<InitializerRelationSolveResult> InitializerRelationSolver::solve(
    std::uint64_t assignmentLimit,
    DeterministicPnrRandomStream *diversificationStream,
    llvm::ArrayRef<PnrIndex> preferredChoices,
    llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
        validateCompleteAssignment) {
  if (rootCardinalityContradiction_) {
    assignmentAttempts_ = 0;
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::ProvenInfeasible,
        "initializer all-different relation has fewer values than members");
  }
  reset();
  auto result = search(assignmentLimit, diversificationStream, preferredChoices,
                       validateCompleteAssignment);
  if (!result)
    return result.takeError();
  if (*result == SearchResult::WorkLimit)
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::WorkLimit,
        "Spatial initializer exhausted its assignment work limit");
  if (*result == SearchResult::Contradiction) {
    if (allDifferentFailureAtInitialPropagation_)
      return llvm::make_error<InitializerRelationSolveFailure>(
          InitializerRelationSolveFailureKind::ProvenInfeasible,
          allDifferentHallFailureMessage());
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::ProvenInfeasible,
        "initializer assignment domain is infeasible");
  }

  InitializerRelationSolveResult solved;
  solved.choices.reserve(domainCounts_.size());
  for (PnrIndex decision = 0; decision < domainCounts_.size(); ++decision)
    solved.choices.push_back(soleChoice(decision));
  solved.assignmentAttempts = assignmentAttempts_;
  return solved;
}

std::size_t InitializerRelationSolver::retainedStorageBytes() const {
  return retainedBytes(decisionChoiceOffsets_) + retainedBytes(activeChoices_) +
         retainedBytes(domainCounts_) + retainedBytes(removalJournal_) +
         retainedBytes(relationQueue_) + retainedBytes(relationPending_) +
         retainedBytes(binaryEqualSupports_) +
         retainedBytes(binaryEqualChoiceValues_) +
         retainedBytes(binaryEqualActiveCounts_) +
         retainedBytes(binaryEqualValueOccurrenceOffsets_) +
         retainedBytes(binaryEqualChoiceOccurrences_) +
         retainedBytes(binaryEqualDepletedValuePending_) +
         retainedBytes(binaryEqualDepletedValueQueue_) +
         retainedBytes(allDifferentSupports_) +
         retainedBytes(allDifferentMembers_) +
         retainedBytes(allDifferentChoiceValues_) +
         retainedBytes(allDifferentActiveChoiceCounts_) +
         retainedBytes(allDifferentForcedDecisionCounts_) +
         retainedBytes(allDifferentValueOccurrenceOffsets_) +
         retainedBytes(allDifferentChoiceOccurrences_) +
         retainedBytes(allDifferentForcedValuePending_) +
         retainedBytes(allDifferentForcedValueQueue_) +
         retainedBytes(allDifferentMemberMatches_) +
         retainedBytes(allDifferentValueMatches_) +
         retainedBytes(allDifferentMemberDistances_) +
         retainedBytes(allDifferentMemberQueue_) +
         retainedBytes(allDifferentValueReachable_) +
         retainedBytes(canonicalActiveChoices_) + retainedBytes(choiceOrder_) +
         retainedBytes(choiceFenwick_) + retainedBytes(completeAssignment_);
}
