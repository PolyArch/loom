#ifndef LOOM_DSE_PROMOTION_H
#define LOOM_DSE_PROMOTION_H

#include "Common/Artifact.h"
#include "DSE/Objective.h"
#include "Evaluation/Evidence.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

class QualityGatePolicyRef final {
public:
  explicit constexpr QualityGatePolicyRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(QualityGatePolicyRef lhs,
                                   QualityGatePolicyRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(QualityGatePolicyRef lhs,
                                   QualityGatePolicyRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

enum class MetricGateComparator : std::uint8_t { LT, LE, EQ, NE, GE, GT };

enum class RequiredFindingState : std::uint8_t { Present, Absent };

enum class GateTruth : std::uint8_t {
  DefinitelyTrue,
  DefinitelyFalse,
  Indeterminate,
};

struct MetricGate final {
  std::uint32_t evidenceObligationTemplate;
  evaluation::MetricRequestOrdinal metricRequest;
  MetricGateComparator comparator;
  evaluation::MetricValue threshold;
};

struct FindingGate final {
  std::uint32_t evidenceObligationTemplate;
  evaluation::FindingRequestOrdinal findingRequest;
  RequiredFindingState requiredState;
};

using QualityGateAtom = std::variant<MetricGate, FindingGate>;

struct QualityGateClause final {
  std::vector<QualityGateAtom> atoms;
};

/// Canonical conjunctive-normal-form quality policy. Empty policies impose no
/// acceptance constraint; clauses are non-empty canonical atom sets.
class QualityGatePolicy final {
public:
  static llvm::Expected<QualityGatePolicy>
  get(std::vector<QualityGateClause> clauses);

  llvm::ArrayRef<QualityGateClause> clauses() const { return clauses_; }
  std::size_t atomCount() const { return atomCount_; }

private:
  QualityGatePolicy(std::vector<QualityGateClause> clauses,
                    std::size_t atomCount)
      : clauses_(std::move(clauses)), atomCount_(atomCount) {}

  std::vector<QualityGateClause> clauses_;
  std::size_t atomCount_;
};

llvm::Expected<GateTruth> evaluateMetricGate(
    evaluation::MetricKind metric, const evaluation::MetricResult &result,
    MetricGateComparator comparator, evaluation::MetricValue threshold);

GateTruth evaluateFindingGate(const evaluation::FindingResult &result,
                              RequiredFindingState requiredState);

/// Truths follow canonical clause and atom order. Any Indeterminate atom makes
/// the whole promotion gate Indeterminate before ordinary CNF evaluation.
llvm::Expected<GateTruth>
evaluateQualityGate(const QualityGatePolicy &policy,
                    llvm::ArrayRef<GateTruth> atomTruths);

class CandidateSet final {
public:
  static llvm::Expected<CandidateSet>
  get(const ArtifactSchemaDescriptor &schema,
      llvm::ArrayRef<ArtifactRootReference> candidates);

  const ArtifactSchemaDescriptor &schema() const { return schema_; }
  llvm::ArrayRef<ArtifactRootReference> candidates() const {
    return candidates_;
  }

private:
  CandidateSet(ArtifactSchemaDescriptor schema,
               std::vector<ArtifactRootReference> candidates)
      : schema_(schema), candidates_(std::move(candidates)) {}

  ArtifactSchemaDescriptor schema_;
  std::vector<ArtifactRootReference> candidates_;
};

struct AllPassingSelection final {};

struct TopKSelection final {
  std::uint32_t totalOrdering;
  std::uint64_t k;
};

struct ParetoSelection final {
  std::vector<std::uint32_t> objectiveDimensions;
};

using CandidateSelectionPolicy =
    std::variant<AllPassingSelection, TopKSelection, ParetoSelection>;

struct CandidateObjectiveVector final {
  ArtifactRootReference candidate;
  ObjectiveVector objective;
};

/// Applies one resolved selection policy to the already gate-qualified subset.
/// Objective vectors are required exactly for TopK and Pareto.
llvm::Expected<std::vector<ArtifactRootReference>> applyCandidateSelection(
    const CandidateSet &candidateSet,
    llvm::ArrayRef<ArtifactRootReference> gateQualifiedCandidates,
    llvm::ArrayRef<CandidateObjectiveVector> objectives,
    const CandidateSelectionPolicy &selection,
    const ObjectiveProgram *objectiveProgram);

enum class ObjectiveDirection : std::uint8_t { Minimize, Maximize };

struct PointMetricTopKSelection final {
  evaluation::MetricRequestOrdinal metricRequest;
  ObjectiveDirection direction;
  std::uint64_t k;
};

struct PromotionEvidence final {
  PromotionEvidence(evaluation::EvaluationRequest request,
                    evaluation::EvaluationEvidence evidence,
                    std::uint32_t obligationTemplate = 0)
      : request(std::move(request)), evidence(std::move(evidence)),
        obligationTemplate(obligationTemplate) {}

  evaluation::EvaluationRequest request;
  evaluation::EvaluationEvidence evidence;
  std::uint32_t obligationTemplate;
};

struct CompletedSelection final {
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
};

struct CompletedNoFeasibleCandidate final {
  std::vector<ArtifactRootReference> satisfiedEvidence;
};

enum class IncompleteSelectionReason : std::uint8_t {
  MissingEvidence,
  UnsupportedEvidence,
  ExecutionFailedEvidence,
  CancelledOrTimeoutEvidence,
  NonComparableEvidence,
  ObjectiveUnavailable,
};

llvm::StringRef toString(IncompleteSelectionReason reason);

struct IncompleteSelection final {
  IncompleteSelectionReason reason;
  ArtifactRootReference candidate;
  std::vector<ArtifactRootReference> retainedEvidence;
};

using PromotionOutcome =
    std::variant<CompletedSelection, CompletedNoFeasibleCandidate,
                 IncompleteSelection>;

/// Applies the common post-acquisition Promotion contract. Candidate
/// association is recovered only through each Evidence Request's distinguished
/// subject role. The objective program is required only by TopK and Pareto.
llvm::Expected<PromotionOutcome>
promoteCandidates(const CandidateSet &candidateSet,
                  evaluation::CaseSubjectRoleRef candidateRole,
                  llvm::ArrayRef<PromotionEvidence> evidence,
                  const QualityGatePolicy &qualityGate,
                  const CandidateSelectionPolicy &selection,
                  const ObjectiveProgram *objectiveProgram,
                  const ArtifactStore &artifactStore);

/// Applies one exact TopK selection over a point-valued Metric result. Each
/// Evidence association is derived from Evidence -> Request -> candidateRole;
/// callers cannot supply a parallel candidate-to-Evidence map.
llvm::Expected<PromotionOutcome>
promoteMetricTopK(const CandidateSet &candidateSet,
                  evaluation::CaseSubjectRoleRef candidateRole,
                  llvm::ArrayRef<PromotionEvidence> evidence,
                  const PointMetricTopKSelection &selection,
                  const ArtifactStore &artifactStore);

/// Applies the same exact TopK gate, but admits only candidates whose selected
/// metric is strictly better than the exact stored-program baseline. If none
/// improves on the baseline, the baseline is the sole selected fallback.
llvm::Expected<PromotionOutcome>
promoteMetricTopKAgainstBaseline(const CandidateSet &candidateSet,
                                 evaluation::CaseSubjectRoleRef candidateRole,
                                 const ArtifactRootReference &baseline,
                                 llvm::ArrayRef<PromotionEvidence> evidence,
                                 const PointMetricTopKSelection &selection,
                                 const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_PROMOTION_H
