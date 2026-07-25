#include "Evaluation/Metric.h"
#include "Common/Artifact.h"
#include "Mapping/Artifact.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::evaluation;

namespace {

static_assert(
    std::is_same_v<loom::mapping::ArtifactIdentity, ArtifactIdentity>);
static_assert(
    std::is_same_v<loom::mapping::EntityReference<loom::mapping::ActorId>,
                   ArtifactReference<loom::mapping::ActorId>>);

void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectErrorContains(const char *test, llvm::Error error,
                         llvm::StringRef expected) {
  if (!error)
    fail(test, "expected an error");
  std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  expectErrorContains(test, value.takeError(), expected);
}

DecimalValue decimal(const char *test, std::int64_t coefficient,
                     std::int64_t exponent) {
  return takeExpected(test, DecimalValue::get(coefficient, exponent));
}

ExactRatio ratio(const char *test, std::uint64_t numerator,
                 std::uint64_t denominator) {
  return takeExpected(test, ExactRatio::get(numerator, denominator));
}

// Every registered metric owns the zero-role whole-case form.
EvaluationScope wholeCaseScope() {
  return EvaluationScope{ScopeFormRef(0), {}};
}

MetricObservation cyclePoint(std::int64_t cycles) {
  return MetricObservation{MetricKind::CycleCount, wholeCaseScope(),
                           UncertaintyKind::ExactWithinModel,
                           PointObservation{MetricValue{IntegerValue(cycles)}}};
}

void metricDescriptorsOwnScopeFormsAndRequestConditions() {
  const MetricDescriptor &cycles = metricDescriptor(MetricKind::CycleCount);
  const MetricDescriptor &period = metricDescriptor(MetricKind::ClockPeriod);

  require(__func__,
          cycles.spelling == "cycle_count" &&
              cycles.valueKind == MetricValueKind::Integer &&
              cycles.dimension == MetricDimension::Cycle &&
              cycles.canonicalUnit == "cycle",
          "the cycle_count descriptor changed");
  require(__func__,
          period.valueKind == MetricValueKind::Decimal &&
              period.valueDomain == MetricValueDomain::Positive &&
              !period.permitsObservationForm(ObservationForm::Censored),
          "the clock_period descriptor changed");

  // A metric owns its own scope forms; the whole-case form has no role.
  require(__func__, cycles.scopeForms.size() == 1,
          "cycle_count lost its owned whole-case scope form");
  require(__func__, cycles.scopeForms[0].roles.empty(),
          "the whole-case scope form gained a role");

  // A metric owns which request-specific conditions apply to it.
  require(__func__,
          cycles.permitsRequestCondition(EvaluationConditionKind::Quantile),
          "cycle_count lost its quantile applicability");
  require(__func__,
          !period.permitsRequestCondition(EvaluationConditionKind::Quantile),
          "clock_period gained a request-specific condition");

  require(__func__,
          takeExpected(__func__, parseMetricKind("runtime")) ==
              MetricKind::Runtime,
          "runtime did not round-trip through the registry");
  expectErrorContains(__func__, parseMetricKind("score"), "unknown MetricKind");
}

void decimalValuesNormalizeCanonically() {
  DecimalValue normalized = decimal(__func__, 1200, -2);
  require(__func__,
          normalized.coefficient() == 12 && normalized.base10Exponent() == 0,
          "trailing coefficient zeros were not moved into the exponent");

  DecimalValue negative = decimal(__func__, -4500, -3);
  require(__func__,
          negative.coefficient() == -45 && negative.base10Exponent() == -1,
          "negative decimal was not normalized");

  DecimalValue zero = decimal(__func__, 0, 99);
  require(__func__, zero.coefficient() == 0 && zero.base10Exponent() == 0,
          "zero must have coefficient zero and exponent zero");

  expectErrorContains(
      __func__, DecimalValue::get(10, std::numeric_limits<std::int64_t>::max()),
      "exponent overflow");
}

void exactRatioNormalizesReducesAndChecksOverflow() {
  // Normalization reduces by greatest common divisor and gives zero the sole
  // encoding 0/1.
  ExactRatio reduced = ratio(__func__, 6, 4);
  require(__func__, reduced.numerator() == 3 && reduced.denominator() == 2,
          "ratio was not reduced by greatest common divisor");
  ExactRatio zero = ratio(__func__, 0, 7);
  require(__func__, zero.numerator() == 0 && zero.denominator() == 1,
          "zero must have the sole encoding 0/1");
  require(__func__, ratio(__func__, 6, 4) == ratio(__func__, 3, 2),
          "equal ratios compared unequal after reduction");

  // A zero denominator is rejected.
  expectErrorContains(__func__, ExactRatio::get(5, 0),
                      "denominator must be positive");

  // reducedModulo normalizes into the half-open range [0, modulus).
  require(__func__,
          takeExpected(
              __func__,
              ratio(__func__, 5, 6).reducedModulo(ratio(__func__, 1, 2))) ==
              ratio(__func__, 1, 3),
          "5/6 mod 1/2 must normalize to 1/3");
  require(__func__,
          takeExpected(
              __func__,
              ratio(__func__, 1, 1).reducedModulo(ratio(__func__, 1, 2))) ==
              ratio(__func__, 0, 1),
          "an exact multiple must normalize to 0/1");
  expectErrorContains(
      __func__, ratio(__func__, 3, 4).reducedModulo(ratio(__func__, 0, 1)),
      "modulus must be positive");

  // Overflow in the checked modulo is reported rather than wrapping. Two
  // coprime denominators near 2^32 force the exact remainder denominator past
  // uint64: (2/4295967341) mod (1/4294967311).
  expectErrorContains(__func__,
                      ratio(__func__, 2, 4295967341ULL)
                          .reducedModulo(ratio(__func__, 1, 4294967311ULL)),
                      "overflow");
}

void observationValidationRejectsIllegalCombinations() {
  if (llvm::Error error = validateMetricObservation(cyclePoint(42)))
    fail(__func__, llvm::toString(std::move(error)));

  MetricObservation wrongValueKind{
      MetricKind::CycleCount, wholeCaseScope(),
      UncertaintyKind::ExactWithinModel,
      PointObservation{MetricValue{decimal(__func__, 42, 0)}}};
  expectErrorContains(__func__, validateMetricObservation(wrongValueKind),
                      "requires integer values");

  MetricObservation descendingInterval{
      MetricKind::ClockPeriod, wholeCaseScope(), UncertaintyKind::Bounded,
      IntervalObservation{MetricValue{decimal(__func__, 11, -10)},
                          MetricValue{decimal(__func__, 9, -10)}}};
  expectErrorContains(__func__, validateMetricObservation(descendingInterval),
                      "lower bound exceeds upper bound");

  MetricObservation censoredWithoutLowerBound{
      MetricKind::CycleCount, wholeCaseScope(), UncertaintyKind::Unknown,
      CensoredObservation{std::nullopt, MetricValue{IntegerValue(10)},
                          CensoredReason::SubjectDidNotComplete}};
  expectErrorContains(__func__,
                      validateMetricObservation(censoredWithoutLowerBound),
                      "requires a lower bound");

  MetricObservation invalidNotApplicable{
      MetricKind::Runtime, wholeCaseScope(), UncertaintyKind::ExactWithinModel,
      NotApplicableObservation{NotApplicableReason::UndefinedForSubject}};
  expectErrorContains(__func__, validateMetricObservation(invalidNotApplicable),
                      "not_applicable requires unknown uncertainty");

  // A scope that the metric's own forms do not define is rejected.
  MetricObservation unknownForm{MetricKind::CycleCount,
                                EvaluationScope{ScopeFormRef(7), {}},
                                UncertaintyKind::ExactWithinModel,
                                PointObservation{MetricValue{IntegerValue(1)}}};
  expectErrorContains(__func__, validateMetricObservation(unknownForm),
                      "unknown scope form");
}

void metricQueriesCanonicalizeByScopeKey() {
  MetricQuery clockWhole{MetricKind::ClockPeriod, wholeCaseScope()};
  MetricQuery cyclesWhole{MetricKind::CycleCount, wholeCaseScope()};
  MetricQuery runtimeWhole{MetricKind::Runtime, wholeCaseScope()};

  std::vector<MetricQuery> forward = takeExpected(
      __func__,
      canonicalizeMetricQueries({runtimeWhole, clockWhole, cyclesWhole}));
  std::vector<MetricQuery> reverse = takeExpected(
      __func__,
      canonicalizeMetricQueries({cyclesWhole, runtimeWhole, clockWhole}));
  require(__func__, forward == reverse,
          "canonical order depends on input order");
  require(__func__,
          forward == std::vector<MetricQuery>{clockWhole, cyclesWhole,
                                              runtimeWhole},
          "canonical query ordering changed");
  expectErrorContains(
      __func__, canonicalizeMetricQueries({clockWhole, clockWhole}),
      "duplicate metric query");
}

void metricTextCarriesTheSharedScope() {
  MetricQuery query{MetricKind::ClockPeriod, wholeCaseScope()};
  const std::string expectedQuery =
      R"({"schema":"evaluation.metric_query","schema_version":"1.0","metric":"clock_period","scope":{"form":0,"targets":[]}})";
  std::string serializedQuery =
      takeExpected(__func__, serializeMetricQuery(query));
  require(__func__, serializedQuery == expectedQuery,
          "canonical metric query bytes changed:\n" + serializedQuery);
  require(__func__,
          takeExpected(__func__, parseMetricQuery(expectedQuery)) == query,
          "metric query text did not round-trip");

  MetricObservation observation{MetricKind::CycleCount, wholeCaseScope(),
                                UncertaintyKind::ExactWithinModel,
                                PointObservation{MetricValue{IntegerValue(7)}}};
  const std::string expectedObservation =
      R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"cycle_count","scope":{"form":0,"targets":[]},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":7}}})";
  std::string serializedObservation =
      takeExpected(__func__, serializeMetricObservation(observation));
  require(__func__, serializedObservation == expectedObservation,
          "canonical metric bytes changed:\n" + serializedObservation);
  require(__func__,
          takeExpected(__func__, parseMetricObservation(expectedObservation)) ==
              observation,
          "metric text did not round-trip");

  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"metric":"cycle_count","schema":"evaluation.metric","schema_version":"1.0","scope":{"form":0,"targets":[]},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":7}}})"),
      "metric JSON is not canonical");
}

} // namespace

int main() {
  metricDescriptorsOwnScopeFormsAndRequestConditions();
  decimalValuesNormalizeCanonically();
  exactRatioNormalizesReducesAndChecksOverflow();
  observationValidationRejectsIllegalCombinations();
  metricQueriesCanonicalizeByScopeKey();
  metricTextCarriesTheSharedScope();
  return 0;
}
