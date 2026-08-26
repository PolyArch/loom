#ifndef LOOM_PNR_INITIALIZERRELATIONSOLVER_H
#define LOOM_PNR_INITIALIZERRELATIONSOLVER_H

#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/PnrIndex.h"
#include "PnR/SpatialPnrWorkLedger.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::pnr::detail {

enum class InitializerRelationKind : std::uint8_t {
  Equal,
  Disjoint,
  Capacity,
};

struct InitializerRelationMemberInput final {
  PnrIndex decision = 0;
  std::vector<PnrIndex> projectedValues;
  std::uint64_t demand = 1;

  InitializerRelationMemberInput() = default;
  InitializerRelationMemberInput(PnrIndex decision,
                                 std::vector<PnrIndex> projectedValues,
                                 std::uint64_t demand = 1)
      : decision(decision), projectedValues(std::move(projectedValues)),
        demand(demand) {}
};

struct InitializerRelationInput final {
  InitializerRelationKind kind = InitializerRelationKind::Equal;
  std::vector<InitializerRelationMemberInput> members;
  std::vector<std::uint64_t> valueCapacities;

  InitializerRelationInput() = default;
  InitializerRelationInput(InitializerRelationKind kind,
                           std::vector<InitializerRelationMemberInput> members,
                           std::vector<std::uint64_t> valueCapacities = {})
      : kind(kind), members(std::move(members)),
        valueCapacities(std::move(valueCapacities)) {}
};

struct InitializerRelationRecord final {
  InitializerRelationKind kind = InitializerRelationKind::Equal;
  PnrIndex memberOffset = 0;
  PnrIndex memberCount = 0;
  PnrIndex valueCapacityOffset = 0;
  PnrIndex valueCapacityCount = 0;
};

struct InitializerRelationMember final {
  PnrIndex decision = 0;
  PnrIndex projectedValueOffset = 0;
  std::uint64_t demand = 1;
};

class InitializerRelationModel final {
public:
  static llvm::Expected<InitializerRelationModel>
  create(std::vector<PnrIndex> decisionChoiceCounts,
         std::vector<InitializerRelationInput> relations);
  llvm::ArrayRef<PnrIndex> decisionChoiceOffsets() const {
    return decisionChoiceOffsets_;
  }
  llvm::ArrayRef<InitializerRelationRecord> relations() const {
    return relations_;
  }
  llvm::ArrayRef<InitializerRelationMember>
  members(const InitializerRelationRecord &relation) const {
    return llvm::ArrayRef(relationMembers_)
        .slice(relation.memberOffset, relation.memberCount);
  }
  PnrIndex projectedValue(const InitializerRelationMember &member,
                          PnrIndex localChoice) const {
    return projectedValues_[member.projectedValueOffset + localChoice];
  }
  llvm::ArrayRef<std::uint64_t>
  valueCapacities(const InitializerRelationRecord &relation) const {
    return llvm::ArrayRef(valueCapacities_)
        .slice(relation.valueCapacityOffset, relation.valueCapacityCount);
  }
  llvm::ArrayRef<PnrIndex> decisionRelationOffsets() const {
    return decisionRelationOffsets_;
  }
  llvm::ArrayRef<PnrIndex> decisionRelations() const {
    return decisionRelations_;
  }
  PnrIndex decisionCount() const {
    return static_cast<PnrIndex>(decisionChoiceOffsets_.size() - 1);
  }
  bool relationSatisfied(PnrIndex relation,
                         llvm::ArrayRef<PnrIndex> choices) const;
  llvm::Error verifyChoices(llvm::ArrayRef<PnrIndex> choices) const;

private:
  std::vector<PnrIndex> decisionChoiceOffsets_;
  std::vector<InitializerRelationRecord> relations_;
  std::vector<InitializerRelationMember> relationMembers_;
  std::vector<PnrIndex> projectedValues_;
  std::vector<std::uint64_t> valueCapacities_;
  std::vector<PnrIndex> decisionRelationOffsets_;
  std::vector<PnrIndex> decisionRelations_;
};

enum class InitializerRelationSolveFailureKind : std::uint8_t {
  ProvenInfeasible,
  FixedRootInfeasible,
  WorkLimit,
};

/// Exact Hall witness produced by the initializer's all-different solver.
/// Members are decision ordinals and values are the relation's projected
/// values, both in canonical order.
struct InitializerRelationHallWitness final {
  PnrIndex relation = 0;
  std::vector<PnrIndex> memberDecisions;
  std::vector<PnrIndex> projectedValues;
};

class InitializerRelationSolveFailure final
    : public llvm::ErrorInfo<InitializerRelationSolveFailure> {
public:
  static char ID;

  InitializerRelationSolveFailure(
      InitializerRelationSolveFailureKind kind, std::string message,
      std::optional<InitializerRelationHallWitness> hallWitness = std::nullopt)
      : kind_(kind), message_(std::move(message)),
        hallWitness_(std::move(hallWitness)) {}

  InitializerRelationSolveFailureKind kind() const { return kind_; }
  const std::optional<InitializerRelationHallWitness> &hallWitness() const {
    return hallWitness_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  InitializerRelationSolveFailureKind kind_;
  std::string message_;
  std::optional<InitializerRelationHallWitness> hallWitness_;
};

struct InitializerRelationSolveResult final {
  std::vector<PnrIndex> choices;
  std::uint64_t assignmentAttempts = 0;
};

class InitializerRelationSolver final {
public:
  explicit InitializerRelationSolver(
      const InitializerRelationModel &model,
      llvm::ArrayRef<PnrIndex> independentChoiceCounts = {},
      SpatialPnrWorkLedgerView workLedger = {});

  llvm::Expected<InitializerRelationSolveResult>
  solveCanonical(std::uint64_t assignmentLimit);
  llvm::Expected<InitializerRelationSolveResult> solveCanonical(
      std::uint64_t assignmentLimit,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithFixedChoices(std::uint64_t assignmentLimit,
                                 llvm::ArrayRef<PnrIndex> fixedChoices);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithReleasedChoices(std::uint64_t assignmentLimit,
                                    llvm::ArrayRef<PnrIndex> fixedChoices,
                                    llvm::ArrayRef<PnrIndex> releasedDecisions);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithReleasedChoices(
      std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
      llvm::ArrayRef<PnrIndex> releasedDecisions,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);
  llvm::Expected<InitializerRelationSolveResult> solveCanonicalWithFixedChoices(
      std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithPreferredChoices(std::uint64_t assignmentLimit,
                                     llvm::ArrayRef<PnrIndex> preferredChoices);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithPreferredChoices(
      std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> preferredChoices,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithFixedAndPreferredChoices(
      std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
      llvm::ArrayRef<PnrIndex> preferredChoices);
  llvm::Expected<InitializerRelationSolveResult>
  solveDiversified(std::uint64_t assignmentLimit,
                   DeterministicPnrRandomStream &diversificationStream);
  llvm::Expected<InitializerRelationSolveResult> solveDiversified(
      std::uint64_t assignmentLimit,
      DeterministicPnrRandomStream &diversificationStream,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);

  std::uint64_t assignmentAttempts() const { return assignmentAttempts_; }
  std::size_t retainedStorageBytes() const;

private:
  struct BinaryEqualSupport final {
    InitializerRelationMember first;
    InitializerRelationMember second;
    std::size_t firstChoiceValueOffset = 0;
    std::size_t secondChoiceValueOffset = 0;
    std::size_t firstCountOffset = 0;
    std::size_t secondCountOffset = 0;
    std::size_t firstValueOccurrenceOffset = 0;
    std::size_t secondValueOccurrenceOffset = 0;
    PnrIndex valueCount = 0;
  };

  struct BinaryEqualDepletedValue final {
    PnrIndex relation = 0;
    PnrIndex value = 0;
    bool first = false;
  };

  struct AllDifferentRelationSupport final {
    std::size_t memberOffset = 0;
    PnrIndex memberCount = 0;
    std::size_t forcedDecisionCountOffset = 0;
    std::size_t valueOccurrenceOffset = 0;
    std::size_t projectedValueOffset = 0;
    std::size_t valueMatchOffset = 0;
    PnrIndex valueCount = 0;
  };

  struct AllDifferentMemberSupport final {
    PnrIndex decision = 0;
    std::size_t choiceValueOffset = 0;
    std::size_t activeChoiceCountOffset = 0;
    PnrIndex activeValueCount = 0;
    PnrIndex soleActiveValue = getInvalidPnrIndex();
  };

  struct AllDifferentChoiceOccurrence final {
    std::size_t member = 0;
    PnrIndex localChoice = 0;
  };

  struct AllDifferentForcedValue final {
    PnrIndex relation = 0;
    PnrIndex value = 0;
  };

  struct RemovedChoice final {
    PnrIndex decision = 0;
    PnrIndex localChoice = 0;
  };

  enum class SearchResult : std::uint8_t {
    Solved,
    Contradiction,
    WorkLimit,
  };

  void reset();
  void clearQueue();
  void enqueueRelation(PnrIndex relation);
  void enqueueDecisionRelations(PnrIndex decision);
  void updateBinaryEqualSupport(PnrIndex decision, PnrIndex localChoice,
                                bool add);
  void enqueueBinaryEqualDepletedValue(PnrIndex relation, PnrIndex value,
                                       bool first);
  bool propagateBinaryEqualDepletedValue(PnrIndex relation, PnrIndex value,
                                         bool first);
  void updateAllDifferentSupport(PnrIndex decision, PnrIndex localChoice,
                                 bool add);
  void enqueueAllDifferentForcedValue(PnrIndex relation, PnrIndex value);
  bool propagateAllDifferentValue(PnrIndex relation, PnrIndex value);
  bool allDifferentMatchingFeasible(PnrIndex relation,
                                    const AllDifferentRelationSupport &support,
                                    bool initialPropagation);
  bool augmentAllDifferentMatching(const AllDifferentRelationSupport &support,
                                   PnrIndex member, PnrIndex shortestLength);
  bool removeChoice(PnrIndex decision, PnrIndex localChoice);
  bool choiceActive(PnrIndex decision, PnrIndex localChoice) const;
  bool relationChoiceSupported(PnrIndex relation, PnrIndex decision,
                               PnrIndex localChoice) const;
  bool equalChoiceSupported(PnrIndex relation, PnrIndex decision,
                            PnrIndex localChoice) const;
  bool disjointChoiceSupported(PnrIndex relation, PnrIndex decision,
                               PnrIndex localChoice) const;
  bool capacityChoiceSupported(PnrIndex relation, PnrIndex decision,
                               PnrIndex localChoice) const;
  bool activeRelationSatisfied(const InitializerRelationRecord &relation) const;
  std::string allDifferentHallFailureMessage() const;
  InitializerRelationHallWitness allDifferentHallFailureWitness() const;
  bool propagate();
  llvm::Expected<SearchResult>
  search(std::uint64_t assignmentLimit,
         DeterministicPnrRandomStream *diversificationStream,
         llvm::ArrayRef<PnrIndex> preferredChoices,
         llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
             validateCompleteAssignment);
  llvm::ArrayRef<PnrIndex>
  buildChoiceOrder(PnrIndex decision,
                   DeterministicPnrRandomStream *diversificationStream,
                   llvm::ArrayRef<PnrIndex> preferredChoices);
  void rollback(std::size_t journalMark);
  PnrIndex soleChoice(PnrIndex decision) const;
  llvm::Expected<InitializerRelationSolveResult>
  solve(std::uint64_t assignmentLimit,
        DeterministicPnrRandomStream *diversificationStream,
        llvm::ArrayRef<PnrIndex> preferredChoices,
        llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
            validateCompleteAssignment);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithFixedAndPreferredChoices(
      std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
      llvm::ArrayRef<PnrIndex> preferredChoices,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);

  const InitializerRelationModel *model_ = nullptr;
  std::vector<PnrIndex> decisionChoiceOffsets_;
  std::vector<std::uint8_t> activeChoices_;
  std::vector<PnrIndex> domainCounts_;
  std::vector<RemovedChoice> removalJournal_;
  std::vector<PnrIndex> relationQueue_;
  std::vector<std::uint8_t> relationPending_;
  std::vector<BinaryEqualSupport> binaryEqualSupports_;
  std::vector<PnrIndex> binaryEqualChoiceValues_;
  std::vector<PnrIndex> binaryEqualActiveCounts_;
  std::vector<std::size_t> binaryEqualValueOccurrenceOffsets_;
  std::vector<PnrIndex> binaryEqualChoiceOccurrences_;
  std::vector<std::uint8_t> binaryEqualDepletedValuePending_;
  std::vector<BinaryEqualDepletedValue> binaryEqualDepletedValueQueue_;
  std::vector<AllDifferentRelationSupport> allDifferentSupports_;
  std::vector<AllDifferentMemberSupport> allDifferentMembers_;
  std::vector<PnrIndex> allDifferentChoiceValues_;
  std::vector<PnrIndex> allDifferentProjectedValues_;
  std::vector<PnrIndex> allDifferentActiveChoiceCounts_;
  std::vector<PnrIndex> allDifferentForcedDecisionCounts_;
  std::vector<std::size_t> allDifferentValueOccurrenceOffsets_;
  std::vector<AllDifferentChoiceOccurrence> allDifferentChoiceOccurrences_;
  std::vector<std::uint8_t> allDifferentForcedValuePending_;
  std::vector<AllDifferentForcedValue> allDifferentForcedValueQueue_;
  std::vector<std::uint8_t> allDifferentMatchingDirty_;
  std::vector<PnrIndex> allDifferentMemberMatches_;
  std::vector<PnrIndex> allDifferentValueMatches_;
  std::vector<PnrIndex> allDifferentMemberDistances_;
  std::vector<PnrIndex> allDifferentMemberQueue_;
  std::vector<std::uint8_t> allDifferentValueReachable_;
  std::vector<PnrIndex> canonicalActiveChoices_;
  std::vector<PnrIndex> choiceOrder_;
  std::vector<PnrIndex> choiceFenwick_;
  std::vector<PnrIndex> completeAssignment_;
  std::size_t queueHead_ = 0;
  std::size_t queueTail_ = 0;
  std::size_t queueCount_ = 0;
  std::uint64_t assignmentAttempts_ = 0;
  SpatialPnrWorkLedgerView workLedger_;
  PnrIndex allDifferentFailureRelation_ = getInvalidPnrIndex();
  PnrIndex allDifferentFailureMatched_ = 0;
  PnrIndex allDifferentFailureMemberCount_ = 0;
  PnrIndex allDifferentFailureValueCount_ = 0;
  std::vector<PnrIndex> allDifferentFailureMemberDecisions_;
  std::vector<PnrIndex> allDifferentFailureProjectedValues_;
  std::uint64_t propagationInvocationCount_ = 0;
  bool allDifferentFailureAtInitialPropagation_ = false;
  bool rootCardinalityContradiction_ = false;
  std::optional<InitializerRelationHallWitness> rootCardinalityFailure_;
};

} // namespace loom::pnr::detail

#endif // LOOM_PNR_INITIALIZERRELATIONSOLVER_H
