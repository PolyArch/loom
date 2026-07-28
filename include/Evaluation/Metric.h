#ifndef LOOM_EVALUATION_METRIC_H
#define LOOM_EVALUATION_METRIC_H

#include "Evaluation/Case.h"
#include "Evaluation/NumericValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::evaluation {

enum class MetricKind {
  CycleCount,
  ClockPeriod,
  Runtime,
  LimitingClockFrequency,
  TotalArea,
  DynamicPower,
  LeakagePower,
};
enum class MetricValueKind { Integer, Decimal };
enum class MetricDimension { Cycle, Time, Frequency, Area, Power };
enum class MetricValueDomain { NonNegative, Positive };

enum class ObservationForm { Point, Interval, Censored, NotApplicable };
enum class UncertaintyKind {
  ExactWithinModel, Bounded, Statistical, Unquantified
};
enum class CensoredReason { SubjectDidNotComplete };
enum class NotApplicableReason { UndefinedForSubject };

constexpr std::uint8_t observationFormMask(ObservationForm form) {
  return std::uint8_t{1} << static_cast<std::uint8_t>(form);
}

constexpr std::uint8_t allObservationFormsMask() {
  return observationFormMask(ObservationForm::Point) |
         observationFormMask(ObservationForm::Interval) |
         observationFormMask(ObservationForm::Censored) |
         observationFormMask(ObservationForm::NotApplicable);
}

struct CensoredReasonPolicy {
  CensoredReason reason;
  bool requiresLowerBound;
  bool permitsUpperBound;
};

struct MetricDescriptor {
  MetricKind kind;
  llvm::StringRef spelling;
  llvm::StringRef semanticDefinition;
  MetricValueKind valueKind;
  MetricDimension dimension;
  llvm::StringRef canonicalUnit;
  MetricValueDomain valueDomain;
  llvm::ArrayRef<ScopeFormDescriptor> scopeForms;
  /// Complete request-specific condition patterns. A metric never turns a
  /// kind-only declaration into a wildcard over case signatures or targets.
  llvm::ArrayRef<ConditionApplicabilityPattern>
      permittedRequestConditionPatterns;
  std::uint8_t permittedObservationForms;
  std::optional<CensoredReasonPolicy> censoredReasonPolicy;

  bool permitsObservationForm(ObservationForm form) const;
};

llvm::ArrayRef<MetricDescriptor> allMetricDescriptors();
const MetricDescriptor &metricDescriptor(MetricKind kind);

llvm::StringRef toString(MetricKind kind);
llvm::StringRef toString(ObservationForm form);
llvm::StringRef toString(UncertaintyKind kind);
llvm::StringRef toString(CensoredReason reason);
llvm::StringRef toString(NotApplicableReason reason);

llvm::Expected<MetricKind> parseMetricKind(llvm::StringRef spelling);
llvm::Expected<ObservationForm> parseObservationForm(llvm::StringRef spelling);
llvm::Expected<UncertaintyKind> parseUncertaintyKind(llvm::StringRef spelling);
llvm::Expected<CensoredReason> parseCensoredReason(llvm::StringRef spelling);
llvm::Expected<NotApplicableReason>
parseNotApplicableReason(llvm::StringRef spelling);

using MetricValue = std::variant<IntegerValue, DecimalValue>;

struct MetricQuery {
  MetricKind metric;
  EvaluationScope scope;

  friend bool operator==(const MetricQuery &lhs, const MetricQuery &rhs) {
    return lhs.metric == rhs.metric && lhs.scope == rhs.scope;
  }
  friend bool operator!=(const MetricQuery &lhs, const MetricQuery &rhs) {
    return !(lhs == rhs);
  }
};

/// Registry-relative validation: the scope resolves against the exact
/// MetricKind's own scope forms. Case-relative anchors, closure, and pattern
/// applicability are checked where the exact case is known.
llvm::Error validateMetricQuery(const MetricQuery &query);

/// Metric-registry-owned admission for one exact scope form against either a
/// case signature contract or a fully resolved exact case. The descriptor
/// context remains error-only. The case context returns the validated
/// reference-cycle basis when the scope form requires one and no basis when it
/// does not; the metric registry owns admission and the case-signature registry
/// owns basis resolution, so the basis is resolved exactly once for the caller
/// to propagate or consume.
llvm::Error validateMetricScopeAdmissibility(
    MetricKind metric, ScopeFormRef form,
    const EvaluationCaseSignatureDescriptor &caseSignature);
llvm::Expected<std::optional<ReferenceCycleBasis>>
validateMetricScopeAdmissibility(
    MetricKind metric, ScopeFormRef form,
    const EvaluationCase &evaluationCase,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore);

/// Canonical query collections sort by registry kind and the complete
/// canonical scope key; exact duplicates are invalid.
llvm::Expected<std::vector<MetricQuery>>
canonicalizeMetricQueries(llvm::ArrayRef<MetricQuery> queries);
llvm::Expected<std::string> serializeMetricQuery(const MetricQuery &query);
llvm::Expected<MetricQuery> parseMetricQuery(llvm::StringRef json);

struct PointObservation {
  MetricValue value;

  friend bool operator==(const PointObservation &lhs,
                         const PointObservation &rhs) {
    return lhs.value == rhs.value;
  }
};

struct IntervalObservation {
  MetricValue lower;
  MetricValue upper;

  friend bool operator==(const IntervalObservation &lhs,
                         const IntervalObservation &rhs) {
    return lhs.lower == rhs.lower && lhs.upper == rhs.upper;
  }
};

struct CensoredObservation {
  std::optional<MetricValue> lower;
  std::optional<MetricValue> upper;
  CensoredReason reason;

  friend bool operator==(const CensoredObservation &lhs,
                         const CensoredObservation &rhs) {
    return lhs.lower == rhs.lower && lhs.upper == rhs.upper &&
           lhs.reason == rhs.reason;
  }
};

struct NotApplicableObservation {
  NotApplicableReason reason;

  friend bool operator==(NotApplicableObservation lhs,
                         NotApplicableObservation rhs) {
    return lhs.reason == rhs.reason;
  }
};

using MetricObservationValue =
    std::variant<PointObservation, IntervalObservation, CensoredObservation,
                 NotApplicableObservation>;

ObservationForm observationForm(const MetricObservationValue &observation);
llvm::Error validateMetricObservationValue(
    MetricKind metric, UncertaintyKind uncertainty,
    const MetricObservationValue &observation);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_METRIC_H
