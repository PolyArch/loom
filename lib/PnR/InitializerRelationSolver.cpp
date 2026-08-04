#include "InitializerRelationSolver.h"

#include "InitializerChoiceOrder.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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
    if (input.members.size() < 2)
      return modelError("a relation has fewer than two members");
    if (input.members.size() > getPnrIndexMax() - model.relationMembers_.size())
      return modelError("relation member count overflows PnrIndex");
    const PnrIndex memberOffset =
        static_cast<PnrIndex>(model.relationMembers_.size());
    for (InitializerRelationMemberInput &member : input.members) {
      if (member.decision >= decisionCount)
        return modelError("a relation member names a foreign decision");
      if (member.projectedValues.size() !=
          decisionChoiceCounts[member.decision])
        return modelError(
            "a relation projection does not cover its decision domain");
      if (member.projectedValues.size() >
          getPnrIndexMax() - model.projectedValues_.size())
        return modelError("projected relation values overflow PnrIndex");
      const PnrIndex projectedValueOffset =
          static_cast<PnrIndex>(model.projectedValues_.size());
      model.projectedValues_.insert(model.projectedValues_.end(),
                                    member.projectedValues.begin(),
                                    member.projectedValues.end());
      model.relationMembers_.push_back({member.decision, projectedValueOffset});
      decisionRelations[member.decision].push_back(relationOrdinal);
    }
    model.relations_.push_back({input.kind, memberOffset,
                                static_cast<PnrIndex>(input.members.size())});
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
      return assignmentError("a hard equality or disjoint relation failed");
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
  canonicalActiveChoices_.resize(choiceCount);
  choiceOrder_.resize(choiceCount);
  choiceFenwick_.resize(choiceCount);
  reset();
}

void InitializerRelationSolver::clearQueue() {
  std::fill(relationPending_.begin(), relationPending_.end(), 0);
  queueHead_ = 0;
  queueTail_ = 0;
  queueCount_ = 0;
}

void InitializerRelationSolver::reset() {
  std::fill(activeChoices_.begin(), activeChoices_.end(), 1);
  for (PnrIndex decision = 0; decision < domainCounts_.size(); ++decision)
    domainCounts_[decision] =
        decisionChoiceOffsets_[decision + 1] - decisionChoiceOffsets_[decision];
  removalJournal_.clear();
  clearQueue();
  assignmentAttempts_ = 0;
  for (PnrIndex relation = 0; relation < model_->relations().size(); ++relation)
    enqueueRelation(relation);
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
           offsets[decision], offsets[decision + 1] - offsets[decision]))
    enqueueRelation(relation);
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
  --domainCounts_[decision];
  removalJournal_.push_back({decision, localChoice});
  enqueueDecisionRelations(decision);
  return domainCounts_[decision] != 0;
}

bool InitializerRelationSolver::equalChoiceSupported(
    const InitializerRelationRecord &relation, PnrIndex decision,
    PnrIndex localChoice) const {
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
    const InitializerRelationRecord &relation, PnrIndex decision,
    PnrIndex localChoice) const {
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

bool InitializerRelationSolver::relationChoiceSupported(
    PnrIndex relation, PnrIndex decision, PnrIndex localChoice) const {
  const InitializerRelationRecord &record = model_->relations()[relation];
  switch (record.kind) {
  case InitializerRelationKind::Equal:
    return equalChoiceSupported(record, decision, localChoice);
  case InitializerRelationKind::Disjoint:
    return disjointChoiceSupported(record, decision, localChoice);
  }
  llvm_unreachable("unknown initializer relation kind");
}

bool InitializerRelationSolver::activeRelationSatisfied(
    const InitializerRelationRecord &relation) const {
  const auto members = model_->members(relation);
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
  while (queueCount_ != 0) {
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
  return true;
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
    ++domainCounts_[removed.decision];
  }
}

InitializerRelationSolver::SearchResult InitializerRelationSolver::search(
    std::uint64_t assignmentLimit,
    DeterministicPnrRandomStream *diversificationStream) {
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
    return SearchResult::Solved;
  }

  const PnrIndex choiceCount =
      decisionChoiceOffsets_[selected + 1] - decisionChoiceOffsets_[selected];
  const llvm::ArrayRef<PnrIndex> choiceOrder =
      buildChoiceOrder(selected, diversificationStream);
  for (PnrIndex selectedChoice : choiceOrder) {
    if (assignmentAttempts_ == assignmentLimit)
      return SearchResult::WorkLimit;
    ++assignmentAttempts_;
    const std::size_t journalMark = removalJournal_.size();
    bool retainedChoice = true;
    for (PnrIndex choice = 0; choice < choiceCount; ++choice)
      if (choice != selectedChoice && choiceActive(selected, choice))
        retainedChoice &= removeChoice(selected, choice);
    const SearchResult result =
        retainedChoice ? search(assignmentLimit, diversificationStream)
                       : SearchResult::Contradiction;
    if (result != SearchResult::Contradiction)
      return result;
    rollback(journalMark);
  }
  return SearchResult::Contradiction;
}

llvm::ArrayRef<PnrIndex> InitializerRelationSolver::buildChoiceOrder(
    PnrIndex decision, DeterministicPnrRandomStream *diversificationStream) {
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
  return llvm::ArrayRef(choiceOrder_).slice(choiceOffset, activeCount);
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonical(std::uint64_t assignmentLimit) {
  return solve(assignmentLimit, nullptr);
}

llvm::Expected<InitializerRelationSolveResult>
InitializerRelationSolver::solveCanonicalWithFixedChoices(
    std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices) {
  if (fixedChoices.size() != domainCounts_.size())
    return assignmentError(
        "fixed choice count does not match the decision domain");
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

  const SearchResult result = search(assignmentLimit, nullptr);
  if (result == SearchResult::WorkLimit)
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::WorkLimit,
        "Spatial initializer exhausted its assignment work limit");
  if (result == SearchResult::Contradiction)
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::FixedRootInfeasible,
        "Spatial initializer fixed choices are infeasible");

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
  return solve(assignmentLimit, &diversificationStream);
}

llvm::Expected<InitializerRelationSolveResult> InitializerRelationSolver::solve(
    std::uint64_t assignmentLimit,
    DeterministicPnrRandomStream *diversificationStream) {
  reset();
  const SearchResult result = search(assignmentLimit, diversificationStream);
  if (result == SearchResult::WorkLimit)
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::WorkLimit,
        "Spatial initializer exhausted its assignment work limit");
  if (result == SearchResult::Contradiction)
    return llvm::make_error<InitializerRelationSolveFailure>(
        InitializerRelationSolveFailureKind::ProvenInfeasible,
        "Spatial initializer relation domain is infeasible");

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
         retainedBytes(canonicalActiveChoices_) + retainedBytes(choiceOrder_) +
         retainedBytes(choiceFenwick_);
}
