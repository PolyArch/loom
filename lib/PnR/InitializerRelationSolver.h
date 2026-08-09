#ifndef LOOM_PNR_INITIALIZERRELATIONSOLVER_H
#define LOOM_PNR_INITIALIZERRELATIONSOLVER_H

#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::pnr::detail {

enum class InitializerRelationKind : std::uint8_t {
  Equal,
  Disjoint,
};

struct InitializerRelationMemberInput final {
  PnrIndex decision = 0;
  std::vector<PnrIndex> projectedValues;
};

struct InitializerRelationInput final {
  InitializerRelationKind kind = InitializerRelationKind::Equal;
  std::vector<InitializerRelationMemberInput> members;
};

struct InitializerRelationRecord final {
  InitializerRelationKind kind = InitializerRelationKind::Equal;
  PnrIndex memberOffset = 0;
  PnrIndex memberCount = 0;
};

struct InitializerRelationMember final {
  PnrIndex decision = 0;
  PnrIndex projectedValueOffset = 0;
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
  std::vector<PnrIndex> decisionRelationOffsets_;
  std::vector<PnrIndex> decisionRelations_;
};

enum class InitializerRelationSolveFailureKind : std::uint8_t {
  ProvenInfeasible,
  FixedRootInfeasible,
  WorkLimit,
};

class InitializerRelationSolveFailure final
    : public llvm::ErrorInfo<InitializerRelationSolveFailure> {
public:
  static char ID;

  InitializerRelationSolveFailure(InitializerRelationSolveFailureKind kind,
                                  std::string message)
      : kind_(kind), message_(std::move(message)) {}

  InitializerRelationSolveFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  InitializerRelationSolveFailureKind kind_;
  std::string message_;
};

struct InitializerRelationSolveResult final {
  std::vector<PnrIndex> choices;
  std::uint64_t assignmentAttempts = 0;
};

class InitializerRelationSolver final {
public:
  explicit InitializerRelationSolver(
      const InitializerRelationModel &model,
      llvm::ArrayRef<PnrIndex> independentChoiceCounts = {});

  llvm::Expected<InitializerRelationSolveResult>
  solveCanonical(std::uint64_t assignmentLimit);
  llvm::Expected<InitializerRelationSolveResult> solveCanonical(
      std::uint64_t assignmentLimit,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);
  llvm::Expected<InitializerRelationSolveResult>
  solveCanonicalWithFixedChoices(std::uint64_t assignmentLimit,
                                 llvm::ArrayRef<PnrIndex> fixedChoices);
  llvm::Expected<InitializerRelationSolveResult> solveCanonicalWithFixedChoices(
      std::uint64_t assignmentLimit, llvm::ArrayRef<PnrIndex> fixedChoices,
      llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
          validateCompleteAssignment);
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
    PnrIndex valueCount = 0;
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
  bool removeChoice(PnrIndex decision, PnrIndex localChoice);
  bool choiceActive(PnrIndex decision, PnrIndex localChoice) const;
  bool relationChoiceSupported(PnrIndex relation, PnrIndex decision,
                               PnrIndex localChoice) const;
  bool equalChoiceSupported(PnrIndex relation, PnrIndex decision,
                            PnrIndex localChoice) const;
  bool disjointChoiceSupported(const InitializerRelationRecord &relation,
                               PnrIndex decision, PnrIndex localChoice) const;
  bool activeRelationSatisfied(const InitializerRelationRecord &relation) const;
  bool propagate();
  llvm::Expected<SearchResult>
  search(std::uint64_t assignmentLimit,
         DeterministicPnrRandomStream *diversificationStream,
         llvm::function_ref<llvm::Expected<bool>(llvm::ArrayRef<PnrIndex>)>
             validateCompleteAssignment);
  llvm::ArrayRef<PnrIndex>
  buildChoiceOrder(PnrIndex decision,
                   DeterministicPnrRandomStream *diversificationStream);
  void rollback(std::size_t journalMark);
  PnrIndex soleChoice(PnrIndex decision) const;
  llvm::Expected<InitializerRelationSolveResult>
  solve(std::uint64_t assignmentLimit,
        DeterministicPnrRandomStream *diversificationStream,
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
  std::vector<PnrIndex> canonicalActiveChoices_;
  std::vector<PnrIndex> choiceOrder_;
  std::vector<PnrIndex> choiceFenwick_;
  std::vector<PnrIndex> completeAssignment_;
  std::size_t queueHead_ = 0;
  std::size_t queueTail_ = 0;
  std::size_t queueCount_ = 0;
  std::uint64_t assignmentAttempts_ = 0;
};

} // namespace loom::pnr::detail

#endif // LOOM_PNR_INITIALIZERRELATIONSOLVER_H
