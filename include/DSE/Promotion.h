#ifndef LOOM_DSE_PROMOTION_H
#define LOOM_DSE_PROMOTION_H

#include "Common/Artifact.h"
#include "Evaluation/Evidence.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

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

enum class ObjectiveDirection : std::uint8_t { Minimize, Maximize };

struct PointMetricTopKSelection final {
  evaluation::MetricRequestOrdinal metricRequest;
  ObjectiveDirection direction;
  std::uint64_t k;
};

struct PromotionEvidence final {
  evaluation::EvaluationRequest request;
  evaluation::EvaluationEvidence evidence;
};

struct CompletedSelection final {
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
};

struct CompletedNoFeasibleCandidate final {};

enum class IncompleteSelectionReason : std::uint8_t {
  MissingEvidence,
  UnsupportedEvidence,
  ExecutionFailedEvidence,
  CancelledOrTimeoutEvidence,
  NonComparableEvidence,
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
