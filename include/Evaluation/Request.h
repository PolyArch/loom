#ifndef LOOM_EVALUATION_REQUEST_H
#define LOOM_EVALUATION_REQUEST_H

#include "Evaluation/Case.h"
#include "Evaluation/Metric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::evaluation {

/// One registered MetricKind paired with one valid EvaluationScope and the
/// canonical set of request-specific conditions under which it is requested.
/// The same query may be requested under distinct conditions; only an exact
/// duplicate request is invalid.
class MetricRequest {
public:
  static llvm::Expected<MetricRequest>
  get(MetricQuery query, llvm::ArrayRef<EvaluationCondition> conditions,
      const EvaluationCase &evaluationCase,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore);

  const MetricQuery &query() const { return query_; }
  llvm::ArrayRef<EvaluationCondition> conditions() const { return conditions_; }

private:
  MetricRequest(MetricQuery query, std::vector<EvaluationCondition> conditions)
      : query_(std::move(query)), conditions_(std::move(conditions)) {}

  MetricQuery query_;
  std::vector<EvaluationCondition> conditions_;
};

/// A removable derived index over exact case facts. It is never serialized
/// into Request or Evidence, and it is not an Artifact identity.
class EvaluationCaseKey {
public:
  using Storage = std::array<std::uint8_t, 32>;

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const EvaluationCaseKey &lhs,
                         const EvaluationCaseKey &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const EvaluationCaseKey &lhs,
                         const EvaluationCaseKey &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit EvaluationCaseKey(Storage bytes) : bytes_(bytes) {}

  friend EvaluationCaseKey baseCaseKey(const EvaluationCase &evaluationCase);
  friend EvaluationCaseKey metricCaseKey(const EvaluationCase &evaluationCase,
                                         const MetricRequest &request);

  Storage bytes_;
};

/// Derived from the exact case-signature reference, the canonical subject
/// bindings, the workload and runtime-input references, and the canonical
/// base conditions. Two model descriptors that reference one exact signature
/// and bind identical case facts therefore derive one key.
EvaluationCaseKey baseCaseKey(const EvaluationCase &evaluationCase);

/// Derived from the base case key, the MetricQuery, and the canonical
/// request-specific conditions, under its own domain separation.
EvaluationCaseKey metricCaseKey(const EvaluationCase &evaluationCase,
                                const MetricRequest &request);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_REQUEST_H
