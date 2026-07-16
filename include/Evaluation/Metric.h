#ifndef LOOM_EVALUATION_METRIC_H
#define LOOM_EVALUATION_METRIC_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::evaluation {

enum class MetricKind { CycleCount, ClockPeriod, Runtime };
enum class MetricValueKind { Integer, Decimal };
enum class MetricDimension { Cycle, Time };
enum class MetricValueDomain { NonNegative, Positive };

enum class ObservationForm { Point, Interval, Censored, NotApplicable };
enum class UncertaintyKind { ExactWithinModel, Bounded, Statistical, Unknown };
enum class CensoredReason { SubjectDidNotComplete };
enum class NotApplicableReason { UndefinedForSubject };

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
  bool permitsEntityScope;
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

class IntegerValue {
public:
  explicit constexpr IntegerValue(std::int64_t value) : value_(value) {}

  constexpr std::int64_t value() const { return value_; }

  friend constexpr bool operator==(IntegerValue lhs, IntegerValue rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(IntegerValue lhs, IntegerValue rhs) {
    return !(lhs == rhs);
  }

private:
  std::int64_t value_;
};

class DecimalValue {
public:
  static llvm::Expected<DecimalValue> get(std::int64_t coefficient,
                                          std::int64_t base10Exponent);

  std::int64_t coefficient() const { return coefficient_; }
  std::int64_t base10Exponent() const { return base10Exponent_; }

  friend bool operator==(DecimalValue lhs, DecimalValue rhs) {
    return lhs.coefficient_ == rhs.coefficient_ &&
           lhs.base10Exponent_ == rhs.base10Exponent_;
  }
  friend bool operator!=(DecimalValue lhs, DecimalValue rhs) {
    return !(lhs == rhs);
  }

private:
  DecimalValue(std::int64_t coefficient, std::int64_t base10Exponent)
      : coefficient_(coefficient), base10Exponent_(base10Exponent) {}

  std::int64_t coefficient_;
  std::int64_t base10Exponent_;
};

using MetricValue = std::variant<IntegerValue, DecimalValue>;

class MetricEntityId {
public:
  explicit constexpr MetricEntityId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(MetricEntityId lhs, MetricEntityId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(MetricEntityId lhs, MetricEntityId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t value_;
};

struct WholeSubjectScope {
  friend constexpr bool operator==(WholeSubjectScope, WholeSubjectScope) {
    return true;
  }
  friend constexpr bool operator!=(WholeSubjectScope, WholeSubjectScope) {
    return false;
  }
};

using MetricEntityReference = ArtifactReference<MetricEntityId>;
using MetricScope = std::variant<WholeSubjectScope, MetricEntityReference>;

struct MetricQuery {
  MetricKind metric;
  MetricScope scope;

  friend bool operator==(const MetricQuery &lhs, const MetricQuery &rhs) {
    return lhs.metric == rhs.metric && lhs.scope == rhs.scope;
  }
  friend bool operator!=(const MetricQuery &lhs, const MetricQuery &rhs) {
    return !(lhs == rhs);
  }
};

llvm::Error validateMetricQuery(const MetricQuery &query);
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

struct MetricObservation {
  MetricKind metric;
  MetricScope scope;
  UncertaintyKind uncertainty;
  MetricObservationValue observation;

  friend bool operator==(const MetricObservation &lhs,
                         const MetricObservation &rhs) {
    return lhs.metric == rhs.metric && lhs.scope == rhs.scope &&
           lhs.uncertainty == rhs.uncertainty &&
           lhs.observation == rhs.observation;
  }
  friend bool operator!=(const MetricObservation &lhs,
                         const MetricObservation &rhs) {
    return !(lhs == rhs);
  }
};

ObservationForm observationForm(const MetricObservation &observation);
llvm::Error validateMetricObservation(const MetricObservation &observation);

llvm::Expected<std::string>
serializeMetricObservation(const MetricObservation &observation);
llvm::Expected<MetricObservation> parseMetricObservation(llvm::StringRef json);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_METRIC_H
