#include "Evaluation/Metric.h"
#include "Common/Artifact.h"
#include "Mapping/Artifact.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::evaluation;

namespace {

static_assert(std::is_same_v<loom::mapping::SchemaVersion, SchemaVersion>);
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

std::string takeErrorMessage(const char *test, llvm::Error error) {
  if (!error)
    fail(test, "expected an error");
  return llvm::toString(std::move(error));
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

MetricObservation cyclePoint(std::int64_t cycles) {
  return MetricObservation{MetricKind::CycleCount, WholeSubjectScope{},
                           UncertaintyKind::ExactWithinModel,
                           PointObservation{MetricValue{IntegerValue(cycles)}}};
}

void sharedArtifactAtomsAreSingleSource() {
  ArtifactIdentity identity({0x01, 0xab});
  require(__func__, !identity.empty(), "identity unexpectedly empty");
  require(__func__, identity.bytes().size() == 2,
          "identity bytes were not preserved");

  loom::mapping::ActorRef mappingReference{identity, loom::mapping::ActorId(7)};
  ArtifactReference<loom::mapping::ActorId> commonReference = mappingReference;
  require(__func__, commonReference.artifact == identity,
          "Mapping reference lost the common artifact identity");
  require(__func__, commonReference.entity == loom::mapping::ActorId(7),
          "Mapping reference lost its typed entity ID");
}

void metricRegistryIsClosedAndTyped() {
  const MetricDescriptor &cycles = metricDescriptor(MetricKind::CycleCount);
  const MetricDescriptor &period = metricDescriptor(MetricKind::ClockPeriod);
  const MetricDescriptor &runtime = metricDescriptor(MetricKind::Runtime);

  require(__func__, cycles.spelling == "cycle_count",
          "cycle_count spelling changed");
  require(__func__, cycles.valueKind == MetricValueKind::Integer,
          "cycle_count must be an integer");
  require(__func__, cycles.dimension == MetricDimension::Cycle,
          "cycle_count dimension changed");
  require(__func__, cycles.canonicalUnit == "cycle",
          "cycle_count unit changed");
  require(__func__, cycles.permitsObservationForm(ObservationForm::Censored),
          "cycle_count must permit censored observations");
  require(__func__, cycles.censoredReasonPolicy.has_value(),
          "cycle_count lacks its censored reason policy");

  require(__func__, period.spelling == "clock_period",
          "clock_period spelling changed");
  require(__func__, period.valueKind == MetricValueKind::Decimal,
          "clock_period must be decimal");
  require(__func__, period.dimension == MetricDimension::Time,
          "clock_period dimension changed");
  require(__func__, period.canonicalUnit == "second",
          "clock_period unit changed");
  require(__func__, !period.permitsObservationForm(ObservationForm::Censored),
          "clock_period must reject censored observations");
  require(__func__, !period.censoredReasonPolicy.has_value(),
          "clock_period has an unsupported censored reason policy");

  require(__func__, runtime.spelling == "runtime", "runtime spelling changed");
  require(__func__, runtime.valueKind == MetricValueKind::Decimal,
          "runtime must be decimal");
  require(__func__, runtime.dimension == MetricDimension::Time,
          "runtime dimension changed");
  require(__func__, runtime.canonicalUnit == "second", "runtime unit changed");
  require(__func__, runtime.censoredReasonPolicy.has_value(),
          "runtime lacks its censored reason policy");
  require(__func__,
          runtime.censoredReasonPolicy->reason ==
                  CensoredReason::SubjectDidNotComplete &&
              runtime.censoredReasonPolicy->requiresLowerBound &&
              !runtime.censoredReasonPolicy->permitsUpperBound,
          "runtime censored reason policy changed");

  std::set<std::string> spellings;
  for (const MetricDescriptor &descriptor : allMetricDescriptors())
    require(__func__, spellings.insert(descriptor.spelling.str()).second,
            "metric registry contains a duplicate spelling");
  require(__func__, spellings.size() == 3,
          "metric registry contains an unapproved metric");

  require(__func__,
          takeExpected(__func__, parseMetricKind("runtime")) ==
              MetricKind::Runtime,
          "runtime did not round-trip through the registry");
  expectErrorContains(__func__, parseMetricKind("score"), "unknown MetricKind");
}

void decimalValuesNormalizeCanonically() {
  DecimalValue normalized = decimal(__func__, 1200, -2);
  require(__func__, normalized.coefficient() == 12,
          "trailing coefficient zeros were not removed");
  require(__func__, normalized.base10Exponent() == 0,
          "decimal exponent did not absorb trailing zeros");

  DecimalValue negative = decimal(__func__, -4500, -3);
  require(__func__, negative.coefficient() == -45,
          "negative decimal was not normalized");
  require(__func__, negative.base10Exponent() == -1,
          "negative decimal exponent was not normalized");

  DecimalValue zero = decimal(__func__, 0, 99);
  require(__func__, zero.coefficient() == 0, "zero coefficient changed");
  require(__func__, zero.base10Exponent() == 0, "zero must have exponent zero");

  require(__func__,
          IntegerValue(std::numeric_limits<std::int64_t>::min()).value() ==
              std::numeric_limits<std::int64_t>::min(),
          "IntegerValue lost INT64_MIN");
  require(__func__,
          IntegerValue(std::numeric_limits<std::int64_t>::max()).value() ==
              std::numeric_limits<std::int64_t>::max(),
          "IntegerValue lost INT64_MAX");
  require(__func__,
          decimal(__func__, std::numeric_limits<std::int64_t>::min(), 0)
                  .coefficient() == std::numeric_limits<std::int64_t>::min(),
          "DecimalValue lost INT64_MIN");
  require(__func__,
          decimal(__func__, std::numeric_limits<std::int64_t>::max(), 0)
                  .coefficient() == std::numeric_limits<std::int64_t>::max(),
          "DecimalValue lost INT64_MAX");

  expectErrorContains(
      __func__, DecimalValue::get(10, std::numeric_limits<std::int64_t>::max()),
      "exponent overflow");
}

void observationValidationRejectsIllegalCombinations() {
  MetricObservation valid = cyclePoint(42);
  if (llvm::Error error = validateMetricObservation(valid))
    fail(__func__, llvm::toString(std::move(error)));

  MetricObservation wrongValueKind{
      MetricKind::CycleCount, WholeSubjectScope{},
      UncertaintyKind::ExactWithinModel,
      PointObservation{MetricValue{decimal(__func__, 42, 0)}}};
  expectErrorContains(__func__, validateMetricObservation(wrongValueKind),
                      "requires integer values");

  MetricObservation descendingInterval{
      MetricKind::ClockPeriod, WholeSubjectScope{}, UncertaintyKind::Bounded,
      IntervalObservation{MetricValue{decimal(__func__, 11, -10)},
                          MetricValue{decimal(__func__, 9, -10)}}};
  expectErrorContains(__func__, validateMetricObservation(descendingInterval),
                      "lower bound exceeds upper bound");

  MetricObservation validCensored{
      MetricKind::Runtime, WholeSubjectScope{}, UncertaintyKind::Unknown,
      CensoredObservation{MetricValue{decimal(__func__, 10, 0)}, std::nullopt,
                          CensoredReason::SubjectDidNotComplete}};
  if (llvm::Error error = validateMetricObservation(validCensored))
    fail(__func__, llvm::toString(std::move(error)));

  MetricObservation clockPeriodCensored{
      MetricKind::ClockPeriod, WholeSubjectScope{}, UncertaintyKind::Unknown,
      CensoredObservation{MetricValue{decimal(__func__, 10, -9)}, std::nullopt,
                          CensoredReason::SubjectDidNotComplete}};
  expectErrorContains(__func__, validateMetricObservation(clockPeriodCensored),
                      "clock_period does not permit censored observations");

  MetricObservation missingCensoredLower{
      MetricKind::CycleCount, WholeSubjectScope{}, UncertaintyKind::Unknown,
      CensoredObservation{std::nullopt, MetricValue{IntegerValue(10)},
                          CensoredReason::SubjectDidNotComplete}};
  expectErrorContains(__func__, validateMetricObservation(missingCensoredLower),
                      "requires a lower bound");

  MetricObservation censoredUpperBound{
      MetricKind::CycleCount, WholeSubjectScope{}, UncertaintyKind::Unknown,
      CensoredObservation{MetricValue{IntegerValue(10)},
                          MetricValue{IntegerValue(20)},
                          CensoredReason::SubjectDidNotComplete}};
  expectErrorContains(__func__, validateMetricObservation(censoredUpperBound),
                      "does not permit an upper bound");

  MetricObservation invalidNotApplicable{
      MetricKind::Runtime, WholeSubjectScope{},
      UncertaintyKind::ExactWithinModel,
      NotApplicableObservation{NotApplicableReason::UndefinedForSubject}};
  expectErrorContains(__func__, validateMetricObservation(invalidNotApplicable),
                      "not_applicable requires unknown uncertainty");

  MetricObservation emptyEntityScope{
      MetricKind::CycleCount,
      MetricEntityReference{ArtifactIdentity(), MetricEntityId(7)},
      UncertaintyKind::ExactWithinModel,
      PointObservation{MetricValue{IntegerValue(1)}}};
  expectErrorContains(__func__, validateMetricObservation(emptyEntityScope),
                      "entity scope requires an artifact identity");
}

void metricQueryCanonicalizationIsInputOrderIndependent() {
  MetricQuery clockWhole{MetricKind::ClockPeriod, WholeSubjectScope{}};
  MetricQuery clockEntityLate{
      MetricKind::ClockPeriod,
      MetricEntityReference{ArtifactIdentity({0x02}), MetricEntityId(1)}};
  MetricQuery clockEntityHighId{
      MetricKind::ClockPeriod,
      MetricEntityReference{ArtifactIdentity({0x01}), MetricEntityId(9)}};
  MetricQuery clockEntityLowId{
      MetricKind::ClockPeriod,
      MetricEntityReference{ArtifactIdentity({0x01}), MetricEntityId(3)}};
  MetricQuery cyclesWhole{MetricKind::CycleCount, WholeSubjectScope{}};
  MetricQuery runtimeWhole{MetricKind::Runtime, WholeSubjectScope{}};

  std::vector<MetricQuery> forward = takeExpected(
      __func__, canonicalizeMetricQueries({cyclesWhole, clockEntityLate,
                                           runtimeWhole, clockEntityHighId,
                                           clockWhole, clockEntityLowId}));
  std::vector<MetricQuery> reverse = takeExpected(
      __func__, canonicalizeMetricQueries({clockEntityLowId, clockWhole,
                                           clockEntityHighId, runtimeWhole,
                                           clockEntityLate, cyclesWhole}));

  require(__func__, forward.size() == 6 && reverse.size() == forward.size(),
          "canonicalization changed the query count");
  for (std::size_t index = 0; index < forward.size(); ++index)
    require(__func__,
            forward[index].metric == reverse[index].metric &&
                forward[index].scope == reverse[index].scope,
            "canonical order depends on input order");

  const std::vector<MetricQuery> expected = {
      clockWhole,      clockEntityLowId, clockEntityHighId,
      clockEntityLate, cyclesWhole,      runtimeWhole};
  for (std::size_t index = 0; index < expected.size(); ++index)
    require(__func__,
            forward[index].metric == expected[index].metric &&
                forward[index].scope == expected[index].scope,
            "canonical query ordering changed");
}

void metricQueryDuplicatesAreRejectedWithoutCollapsingScopes() {
  MetricQuery whole{MetricKind::CycleCount, WholeSubjectScope{}};
  MetricQuery firstEntity{
      MetricKind::CycleCount,
      MetricEntityReference{ArtifactIdentity({0x01}), MetricEntityId(1)}};
  MetricQuery secondEntity{
      MetricKind::CycleCount,
      MetricEntityReference{ArtifactIdentity({0x01}), MetricEntityId(2)}};

  std::vector<MetricQuery> distinct = takeExpected(
      __func__, canonicalizeMetricQueries({secondEntity, whole, firstEntity}));
  require(__func__, distinct.size() == 3,
          "distinct typed scopes were collapsed");
  expectErrorContains(__func__,
                      canonicalizeMetricQueries(
                          {whole, firstEntity, secondEntity, firstEntity}),
                      "duplicate metric query");
}

void metricQueryUsesSharedScopeValidation() {
  MetricQuery query{
      MetricKind::CycleCount,
      MetricEntityReference{ArtifactIdentity(), MetricEntityId(7)}};
  MetricObservation observation{
      MetricKind::CycleCount,
      MetricEntityReference{ArtifactIdentity(), MetricEntityId(7)},
      UncertaintyKind::ExactWithinModel,
      PointObservation{MetricValue{IntegerValue(1)}}};

  const std::string queryError =
      takeErrorMessage(__func__, validateMetricQuery(query));
  const std::string observationError =
      takeErrorMessage(__func__, validateMetricObservation(observation));
  require(__func__, queryError == observationError,
          "query and observation scope validation diverged");
  require(__func__,
          llvm::StringRef(queryError)
              .contains("entity scope requires an artifact identity"),
          "invalid entity scope produced an unclear error");
}

void canonicalJsonIsStableAndStrict() {
  MetricObservation observation{
      MetricKind::ClockPeriod,
      MetricEntityReference{
          ArtifactIdentity({0x01, 0xab}),
          MetricEntityId(std::numeric_limits<std::uint64_t>::max())},
      UncertaintyKind::Bounded,
      IntervalObservation{MetricValue{decimal(__func__, 9, -10)},
                          MetricValue{decimal(__func__, 11, -10)}}};

  const std::string expected =
      R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"clock_period","scope":{"kind":"entity","artifact":"01ab","entity_id":18446744073709551615},"uncertainty":"bounded","observation":{"form":"interval","lower":{"kind":"decimal","coefficient":9,"base10_exponent":-10},"upper":{"kind":"decimal","coefficient":11,"base10_exponent":-10}}})";
  std::string serialized =
      takeExpected(__func__, serializeMetricObservation(observation));
  require(__func__, serialized == expected,
          "canonical JSON bytes changed:\n" + serialized);

  MetricObservation parsed =
      takeExpected(__func__, parseMetricObservation(expected));
  require(__func__, parsed == observation, "canonical JSON did not round-trip");
  require(__func__,
          takeExpected(__func__, serializeMetricObservation(parsed)) ==
              expected,
          "round-trip serialization was not byte-stable");

  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.1","metric":"cycle_count","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}}})"),
      "unsupported evaluation.metric version");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.other","schema_version":"1.0","metric":"cycle_count","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}}})"),
      "unsupported metric schema");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"cycle_count","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}},"score":1})"),
      "unknown field 'score'");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}}})"),
      "field 'metric' must be a string");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"cycle_count","metric":"cycle_count","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}}})"),
      "metric JSON is not canonical");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"cycle_count","scope":{"kind":"whole_subject","path":"graph/0"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}}})"),
      "metric scope has unknown field 'path'");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"metric":"cycle_count","schema":"evaluation.metric","schema_version":"1.0","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}}})"),
      "metric JSON is not canonical");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"clock_period","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"decimal","coefficient":120,"base10_exponent":-2}}})"),
      "decimal is not canonical");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"clock_period","scope":{"kind":"whole_subject"},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"decimal","coefficient":1.0,"base10_exponent":-9}}})"),
      "metric JSON is not canonical");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"cycle_count","scope":{"kind":"entity","artifact":"01AB","entity_id":7},"uncertainty":"exact_within_model","observation":{"form":"point","value":{"kind":"integer","value":1}}})"),
      "artifact identity must use lowercase hexadecimal");
  expectErrorContains(
      __func__,
      parseMetricObservation(
          R"({"schema":"evaluation.metric","schema_version":"1.0","metric":"cycle_count","scope":{"kind":"whole_subject"},"uncertainty":"unknown","observation":{"form":"not_applicable","reason":"subject_did_not_complete"}})"),
      "unknown NotApplicableReason");

  std::string pointJson =
      takeExpected(__func__, serializeMetricObservation(cyclePoint(7)));
  require(__func__, pointJson.find("score") == std::string::npos,
          "metric schema contains score");
  require(__func__, pointJson.find("objective") == std::string::npos,
          "metric schema contains objective");
  require(__func__, pointJson.find("acceptance") == std::string::npos,
          "metric schema contains acceptance");
}

} // namespace

int main() {
  sharedArtifactAtomsAreSingleSource();
  metricRegistryIsClosedAndTyped();
  decimalValuesNormalizeCanonically();
  observationValidationRejectsIllegalCombinations();
  metricQueryCanonicalizationIsInputOrderIndependent();
  metricQueryDuplicatesAreRejectedWithoutCollapsingScopes();
  metricQueryUsesSharedScopeValidation();
  canonicalJsonIsStableAndStrict();
  return 0;
}
