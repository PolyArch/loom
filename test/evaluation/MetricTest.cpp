#include "Evaluation/Metric.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

using namespace loom::evaluation;

namespace {

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

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

ExactRatio ratio(const char *test, std::uint64_t numerator,
                 std::uint64_t denominator) {
  return takeExpected(test, ExactRatio::get(numerator, denominator));
}

void exactRatioNormalizesAndChecksArithmetic() {
  const ExactRatio reduced = ratio(__func__, 6, 4);
  require(__func__, reduced.numerator() == 3 && reduced.denominator() == 2,
          "ratio was not reduced by greatest common divisor");
  const ExactRatio zero = ratio(__func__, 0, 7);
  require(__func__, zero.numerator() == 0 && zero.denominator() == 1,
          "zero must have the sole encoding 0/1");
  expectErrorContains(__func__, ExactRatio::get(5, 0),
                      "denominator must be positive");

  require(__func__,
          takeExpected(
              __func__,
              ratio(__func__, 5, 6).reducedModulo(ratio(__func__, 1, 2))) ==
              ratio(__func__, 1, 3),
          "5/6 mod 1/2 must normalize to 1/3");
  expectErrorContains(__func__,
                      ratio(__func__, 2, 4295967341ULL)
                          .reducedModulo(ratio(__func__, 1, 4294967311ULL)),
                      "overflow");
}

void builtInMetricsOwnWholeCaseFormZero() {
  for (MetricKind metric :
       {MetricKind::CycleCount, MetricKind::ClockPeriod, MetricKind::Runtime,
        MetricKind::LimitingClockFrequency, MetricKind::TotalArea,
        MetricKind::DynamicPower, MetricKind::LeakagePower,
        MetricKind::MaximumVoltageDrop}) {
    const MetricDescriptor &descriptor = metricDescriptor(metric);
    require(__func__, descriptor.scopeForms.size() == 1,
            "built-in metric lost its sole whole-case scope form");
    if (llvm::Error error = validateMetricQuery(
            MetricQuery{metric, EvaluationScope{ScopeFormRef(0), {}}}))
      fail(__func__, llvm::toString(std::move(error)));
  }

  expectErrorContains(
      __func__,
      llvm::Expected<std::vector<MetricQuery>>(canonicalizeMetricQueries(
          {{MetricKind::CycleCount, EvaluationScope{ScopeFormRef(1), {}}}})),
      "unknown scope form ordinal");

  require(__func__,
          metricDescriptor(MetricKind::CycleCount)
                  .scopeForms[0]
                  .referenceCycleRequirement ==
              ReferenceCycleRequirement::ExactCaseUniqueReferenceCycle,
          "cycle count lost its registry-owned reference-cycle requirement");
  require(__func__,
          metricDescriptor(MetricKind::ClockPeriod)
                  .scopeForms[0]
                  .referenceCycleRequirement ==
              ReferenceCycleRequirement::ExactCaseUniqueReferenceCycle,
          "clock period lost its registry-owned reference-cycle requirement");
  require(__func__,
          metricDescriptor(MetricKind::Runtime)
                  .scopeForms[0]
                  .referenceCycleRequirement ==
              ReferenceCycleRequirement::NotRequired,
          "runtime unexpectedly requires a reference-cycle basis");

  const MetricQuery query{MetricKind::CycleCount,
                          EvaluationScope{ScopeFormRef(0), {}}};
  const std::string canonical =
      takeExpected(__func__, serializeMetricQuery(query));
  require(__func__,
          canonical ==
              "{\"schema\":\"evaluation.metric_query\",\"schema_version\":"
              "\"1.0\",\"metric\":\"cycle_count\",\"scope\":{\"form\":0,"
              "\"targets\":[]}}",
          "metric query wire changed");
  require(__func__,
          takeExpected(__func__, parseMetricQuery(canonical)) == query,
          "metric query payload did not roundtrip");

  struct PhysicalMetricExpectation {
    MetricKind kind;
    llvm::StringRef spelling;
    MetricDimension dimension;
    llvm::StringRef unit;
  };
  const PhysicalMetricExpectation physicalMetrics[] = {
      {MetricKind::LimitingClockFrequency, "limiting_clock_frequency",
       MetricDimension::Frequency, "hertz"},
      {MetricKind::TotalArea, "total_area", MetricDimension::Area,
       "square_meter"},
      {MetricKind::DynamicPower, "dynamic_power", MetricDimension::Power,
       "watt"},
      {MetricKind::LeakagePower, "leakage_power", MetricDimension::Power,
       "watt"},
      {MetricKind::MaximumVoltageDrop, "maximum_voltage_drop",
       MetricDimension::Voltage, "volt"},
  };
  for (const PhysicalMetricExpectation &expected : physicalMetrics) {
    const MetricDescriptor &descriptor = metricDescriptor(expected.kind);
    require(__func__, descriptor.spelling == expected.spelling,
            "physical metric changed its canonical spelling");
    require(__func__,
            descriptor.dimension == expected.dimension &&
                descriptor.canonicalUnit == expected.unit,
            "physical metric changed its dimension or canonical unit");
    require(__func__,
            descriptor.scopeForms[0].referenceCycleRequirement ==
                ReferenceCycleRequirement::NotRequired,
            "whole-case physical metric unexpectedly requires one cycle");
    require(__func__,
            takeExpected(__func__, parseMetricKind(expected.spelling)) ==
                expected.kind,
            "physical metric spelling did not roundtrip through the registry");
  }
}

void unquantifiedUncertaintyHasOneCanonicalSpelling() {
  require(__func__, toString(UncertaintyKind::Unquantified) == "unquantified",
          "unquantified uncertainty changed its canonical spelling");
  require(__func__,
          takeExpected(__func__, parseUncertaintyKind("unquantified")) ==
              UncertaintyKind::Unquantified,
          "unquantified uncertainty did not roundtrip");
  expectErrorContains(__func__, parseUncertaintyKind("unknown"),
                      "unknown UncertaintyKind");
}

} // namespace

int main() {
  exactRatioNormalizesAndChecksArithmetic();
  builtInMetricsOwnWholeCaseFormZero();
  unquantifiedUncertaintyHasOneCanonicalSpelling();
  return 0;
}
