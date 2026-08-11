#include "ADG/BuiltinDescriptor.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CalibratedFpa.h"
#include "Evaluation/Models/FabricLowConfidence.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::evaluation;
using namespace loom::evaluation::models;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "fast FPA model test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

void requireError(llvm::Error error, llvm::StringRef expected) {
  if (!error)
    fail("expected an error containing '" + expected + "'");
  const std::string message = llvm::toString(std::move(error));
  if (message.find(expected.str()) == std::string::npos)
    fail("unexpected error: " + message);
}

template <typename T>
void requireExpectedError(llvm::Expected<T> value, llvm::StringRef expected) {
  if (value)
    fail("expected an error containing '" + expected + "'");
  requireError(value.takeError(), expected);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error =
        llvm::sys::fs::createUniqueDirectory("loom-fast-fpa-model", root_);
    if (error)
      fail("cannot create test directory: " + error.message());
    blobs_ = root_;
    llvm::sys::path::append(blobs_, "blobs");
    error = llvm::sys::fs::create_directory(blobs_);
    if (error)
      fail("cannot create BlobStore directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(root_); }

  llvm::StringRef root() const { return root_; }
  llvm::StringRef blobs() const { return blobs_; }

private:
  llvm::SmallString<128> root_;
  llvm::SmallString<128> blobs_;
};

struct Fixture final {
  TemporaryDirectory directory;
  ArtifactStore artifacts;
  BlobStore blobs;

  Fixture() : artifacts(directory.root()), blobs(directory.blobs()) {
    requireSuccess(registerProductionEvaluationRegistry());
  }
};

DecimalValue decimal(std::int64_t coefficient, std::int64_t exponent = 0) {
  return take(DecimalValue::get(coefficient, exponent));
}

ExactRatio ratio(std::uint64_t numerator, std::uint64_t denominator) {
  return take(ExactRatio::get(numerator, denominator));
}

const CompletedCandidateGeneratorResult &
completedGenerator(const CandidateGeneratorProviderResult &result) {
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&result.outcome);
  if (!completed)
    fail("Fabric template generator did not complete");
  return *completed;
}

loom::fabric::FinalizedFabricRoot
generateSystem(Fixture &fixture, adg::BuiltinTargetPreset preset) {
  auto config = take(resolveFabricTemplateConfig(preset));
  auto inputs = take(bindFabricTemplateCandidateGeneratorInputs());
  auto binding = take(resolveFabricTemplateCandidateGeneratorBinding(config));
  auto result = take(invokeCandidateGenerator(
      inputs, binding, fixture.artifacts, fixture.blobs));
  const auto &outputs = completedGenerator(result).outputBindings;
  require(outputs.size() == 1 && outputs.front().artifacts.size() == 1,
          "Fabric template generator returned the wrong output shape");
  return take(loom::fabric::importEntireFabricRoot(
      outputs.front().artifacts.front(), fixture.artifacts));
}

SubjectTargetRef rootTarget(const ArtifactRootReference &fabric) {
  return {fabricHardwareAnalysisSubjectRole(), fabric, fabric};
}

std::vector<EvaluationCondition>
activityConditions(const ArtifactRootReference &fabric,
                   std::uint64_t transitionNumerator) {
  const SubjectTargetRef target = rootTarget(fabric);
  return {EvaluationCondition{ActivityBindingCondition{
      target, ExplicitAssumptionSource{target, ratio(1, 2),
                                       ratio(transitionNumerator, 10)}}}};
}

std::vector<EvaluationCondition> physicalConditions(
    const ArtifactRootReference &fabric,
    const platform::FinalizedImplementationPlatform &implementationPlatform,
    bool includeActivity = true) {
  const SubjectTargetRef target = rootTarget(fabric);
  std::vector<EvaluationCondition> conditions = {
      EvaluationCondition{
          ProcessCornerCondition{target,
                                 {implementationPlatform.reference().artifact,
                                  platform::TechnologyCornerId(0)}}},
      EvaluationCondition{SupplyVoltageCondition{target, decimal(9, -1)}},
      EvaluationCondition{TemperatureCondition{target, decimal(3, 2)}},
      EvaluationCondition{RequiredClockPeriodCondition{target, decimal(2, -9)}},
  };
  if (includeActivity)
    conditions.push_back(EvaluationCondition{ActivityBindingCondition{
        target, ExplicitAssumptionSource{target, ratio(1, 2), ratio(1, 10)}}});
  return conditions;
}

const CompletedEvidence &completedEvidence(const EvaluationEvidence &evidence) {
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  if (!completed)
    fail("Evaluation did not complete");
  return *completed;
}

DecimalValue metricValue(const EvaluationRequest &request,
                         const EvaluationEvidence &evidence,
                         MetricKind requested) {
  const CompletedEvidence &completed = completedEvidence(evidence);
  for (std::size_t index = 0; index != request.metricRequests().size();
       ++index) {
    if (request.metricRequests()[index].query().metric != requested)
      continue;
    const auto *point = std::get_if<PointObservation>(
        &completed.metricResults[index].observation);
    const auto *value =
        point ? std::get_if<DecimalValue>(&point->value) : nullptr;
    if (!value)
      fail("requested FPA metric is not a Decimal Point");
    return *value;
  }
  fail("requested FPA metric is absent");
}

EvaluationEvidence
evaluatePrepared(const PreparedFabricLowConfidenceEvaluation &prepared,
                 Fixture &fixture) {
  return take(evaluateRequest(prepared.request, prepared.resolution,
                              fixture.artifacts, fixture.blobs));
}

void lowConfidencePhysicalBehavior(
    Fixture &fixture, const loom::fabric::FinalizedFabricRoot &small,
    const loom::fabric::FinalizedFabricRoot &large) {
  const std::array staticMetrics = {MetricKind::LimitingClockFrequency,
                                    MetricKind::TotalArea,
                                    MetricKind::LeakagePower};
  auto smallStatic = take(prepareFabricLowConfidenceEvaluation(
      small.reference(), {}, staticMetrics, defaultResolvedConfig(),
      fixture.artifacts, fixture.blobs));
  auto largeStatic = take(prepareFabricLowConfidenceEvaluation(
      large.reference(), {}, staticMetrics, defaultResolvedConfig(),
      fixture.artifacts, fixture.blobs));
  EvaluationEvidence smallStaticEvidence =
      evaluatePrepared(smallStatic, fixture);
  EvaluationEvidence largeStaticEvidence =
      evaluatePrepared(largeStatic, fixture);
  require(
      compareDecimalValue(metricValue(largeStatic.request, largeStaticEvidence,
                                      MetricKind::TotalArea),
                          metricValue(smallStatic.request, smallStaticEvidence,
                                      MetricKind::TotalArea)) > 0,
      "larger concrete System did not increase low-confidence area");
  require(
      compareDecimalValue(metricValue(largeStatic.request, largeStaticEvidence,
                                      MetricKind::LeakagePower),
                          metricValue(smallStatic.request, smallStaticEvidence,
                                      MetricKind::LeakagePower)) > 0,
      "larger concrete System did not increase low-confidence leakage");

  auto missingActivity = take(prepareFabricLowConfidenceEvaluation(
      small.reference(), {}, {MetricKind::DynamicPower},
      defaultResolvedConfig(), fixture.artifacts, fixture.blobs));
  EvaluationEvidence missingEvidence =
      evaluatePrepared(missingActivity, fixture);
  require(
      std::holds_alternative<UnsupportedEvidence>(missingEvidence.outcome()),
      "missing activity produced a synthetic dynamic-power value");

  auto lowActivity = take(prepareFabricLowConfidenceEvaluation(
      small.reference(), activityConditions(small.reference(), 1),
      {MetricKind::DynamicPower}, defaultResolvedConfig(), fixture.artifacts,
      fixture.blobs));
  auto highActivity = take(prepareFabricLowConfidenceEvaluation(
      small.reference(), activityConditions(small.reference(), 2),
      {MetricKind::DynamicPower}, defaultResolvedConfig(), fixture.artifacts,
      fixture.blobs));
  EvaluationEvidence lowEvidence = evaluatePrepared(lowActivity, fixture);
  EvaluationEvidence highEvidence = evaluatePrepared(highActivity, fixture);
  require(compareDecimalValue(metricValue(highActivity.request, highEvidence,
                                          MetricKind::DynamicPower),
                              metricValue(lowActivity.request, lowEvidence,
                                          MetricKind::DynamicPower)) > 0,
          "greater exact activity did not increase dynamic power");
}

EvaluationCase
recoverCase(const PreparedFabricLowConfidenceEvaluation &prepared,
            Fixture &fixture) {
  return take(EvaluationCase::get(
      fabricHardwareAnalysisCaseSignatureRef(),
      prepared.request.subjectBindings(), std::nullopt, std::nullopt,
      prepared.request.baseConditions(), prepared.resolution, fixture.artifacts,
      fixture.blobs));
}

std::vector<MetricRequest>
fpaMetricRequests(const EvaluationCase &evaluationCase,
                  const CaseArtifactResolution &resolution, Fixture &fixture,
                  bool includeDynamic = true) {
  std::vector<MetricRequest> metrics;
  for (MetricKind kind :
       {MetricKind::LimitingClockFrequency, MetricKind::TotalArea,
        MetricKind::DynamicPower, MetricKind::LeakagePower}) {
    if (!includeDynamic && kind == MetricKind::DynamicPower)
      continue;
    metrics.push_back(take(
        MetricRequest::get({kind, EvaluationScope{ScopeFormRef(0), {}}}, {},
                           evaluationCase, resolution, fixture.artifacts)));
  }
  return metrics;
}

FinalizedModelParameterBundle trainBundle(const FpaFeatureView &features,
                                          Fixture &fixture) {
  const FpaMetricPredictionView observation{decimal(5, 8), decimal(2, -6),
                                            decimal(3, -3), decimal(4, -4)};
  std::vector<FpaTrainingEvidenceSample> rows;
  for (std::uint8_t group = 1; group != 4; ++group)
    rows.push_back({features, observation, {0x51, 0x52}, {group}});
  FpaGbdtParameters parameters = take(
      trainFpaGbdtParameters(rows, FpaGbdtTrainingConfig{17, 3, 2, 1, 1, 2}));
  OwnerValue owner = OwnerValue::get(std::move(parameters));
  return take(finalizeModelParameterBundle(
      fpaModelParameterContractRef(), owner, fixture.artifacts, fixture.blobs));
}

EvaluationRequest calibratedRequest(
    const EvaluationCase &evaluationCase,
    const CaseArtifactResolution &resolution,
    const FinalizedModelParameterBundle &bundle,
    const platform::FinalizedImplementationPlatform &implementationPlatform,
    Fixture &fixture, bool includeDynamic = true) {
  auto binding = take(ResolvedModelBinding::project(
      fabricCalibratedFpaModelDescriptorRef(),
      {{ModelInputSlotRef(0), {bundle.reference()}},
       {ModelInputSlotRef(1), {implementationPlatform.reference()}}},
      defaultResolvedConfig()));
  auto request = take(EvaluationRequest::get(
      evaluationCase,
      fpaMetricRequests(evaluationCase, resolution, fixture, includeDynamic),
      {}, std::move(binding), 0, resolution, fixture.artifacts, fixture.blobs));
  take(publishEvaluationRequest(request, fixture.artifacts));
  return request;
}

void calibratedPredictionBehavior(
    Fixture &fixture, const loom::fabric::FinalizedFabricRoot &small,
    const loom::fabric::FinalizedFabricRoot &large) {
  auto implementationPlatform = take(platform::finalizeImplementationPlatform(
      {platform::AsicTarget{"fast-fpa-test", "2026.08"}, {"typical"}},
      fixture.artifacts));
  auto smallPrepared = take(prepareFabricLowConfidenceEvaluation(
      small.reference(),
      physicalConditions(small.reference(), implementationPlatform),
      {MetricKind::TotalArea}, defaultResolvedConfig(), fixture.artifacts,
      fixture.blobs));
  EvaluationCase smallCase = recoverCase(smallPrepared, fixture);
  OwnerValue projected = take(projectModelFeatures(
      fpaModelParameterContractRef(), smallCase, smallPrepared.resolution,
      fixture.artifacts, fixture.blobs));
  const auto *features = projected.getIf<FpaFeatureView>();
  require(features, "FPA contract did not project its typed feature view");
  FinalizedModelParameterBundle bundle = trainBundle(*features, fixture);

  EvaluationRequest request =
      calibratedRequest(smallCase, smallPrepared.resolution, bundle,
                        implementationPlatform, fixture);
  EvaluationEvidence evidence = take(evaluateRequest(
      request, smallPrepared.resolution, fixture.artifacts, fixture.blobs));
  const CompletedEvidence &completed = completedEvidence(evidence);
  for (const MetricResult &result : completed.metricResults)
    require(result.uncertainty == UncertaintyKind::Unquantified &&
                result.calibrationInputSlots ==
                    std::vector<ModelInputSlotRef>{ModelInputSlotRef(0)},
            "calibrated prediction lost parameter provenance or uncertainty");

  auto largePrepared = take(prepareFabricLowConfidenceEvaluation(
      large.reference(),
      physicalConditions(large.reference(), implementationPlatform),
      {MetricKind::TotalArea}, defaultResolvedConfig(), fixture.artifacts,
      fixture.blobs));
  EvaluationCase largeCase = recoverCase(largePrepared, fixture);
  EvaluationRequest outside =
      calibratedRequest(largeCase, largePrepared.resolution, bundle,
                        implementationPlatform, fixture);
  EvaluationEvidence outsideEvidence = take(evaluateRequest(
      outside, largePrepared.resolution, fixture.artifacts, fixture.blobs));
  require(
      std::holds_alternative<UnsupportedEvidence>(outsideEvidence.outcome()),
      "out-of-envelope calibrated FPA returned an extrapolation");

  auto missingPrepared = take(prepareFabricLowConfidenceEvaluation(
      small.reference(),
      physicalConditions(small.reference(), implementationPlatform, false),
      {MetricKind::TotalArea}, defaultResolvedConfig(), fixture.artifacts,
      fixture.blobs));
  EvaluationCase missingCase = recoverCase(missingPrepared, fixture);
  EvaluationRequest missing =
      calibratedRequest(missingCase, missingPrepared.resolution, bundle,
                        implementationPlatform, fixture, true);
  EvaluationEvidence missingEvidence = take(evaluateRequest(
      missing, missingPrepared.resolution, fixture.artifacts, fixture.blobs));
  require(
      std::holds_alternative<UnsupportedEvidence>(missingEvidence.outcome()),
      "calibrated dynamic power accepted missing activity");

  auto foreignPlatform = take(platform::finalizeImplementationPlatform(
      {platform::AsicTarget{"foreign-fast-fpa", "2026.08"}, {"typical"}},
      fixture.artifacts));
  auto foreignBinding = take(ResolvedModelBinding::project(
      fabricCalibratedFpaModelDescriptorRef(),
      {{ModelInputSlotRef(0), {bundle.reference()}},
       {ModelInputSlotRef(1), {foreignPlatform.reference()}}},
      defaultResolvedConfig()));
  requireExpectedError(
      EvaluationRequest::get(
          smallCase,
          fpaMetricRequests(smallCase, smallPrepared.resolution, fixture,
                            false),
          {}, std::move(foreignBinding), 0, smallPrepared.resolution,
          fixture.artifacts, fixture.blobs),
      "foreign ImplementationPlatform");
}

void exactCalibrationKernel() {
  const std::vector<std::pair<DecimalValue, DecimalValue>> samples = {
      {decimal(0), decimal(0)},
      {decimal(2), decimal(1)},
      {decimal(1), decimal(0)},
  };
  const DecimalValue median =
      take(calculateFpaPredictionErrorQuantile(samples, ratio(1, 2)));
  require(median == decimal(666666666666666667LL, -18),
          "exact median did not use symmetric error and ties-to-even rounding");
  const DecimalValue p90 =
      take(calculateFpaPredictionErrorQuantile(samples, ratio(9, 10)));
  require(p90 == decimal(2), "nearest-rank P90 selected the wrong sample");

  const std::array huge = {std::pair{decimal(1, 1000), decimal(1, 999)}};
  const DecimalValue hugeError =
      take(calculateFpaPredictionErrorQuantile(huge, ratio(0, 1)));
  require(hugeError == decimal(1636363636363636364LL, -18),
          "large decimal exponents lost exact calibration ordering");
}

} // namespace

int main() {
  Fixture fixture;
  const loom::fabric::FinalizedFabricRoot small =
      generateSystem(fixture, adg::BuiltinTargetPreset::Small);
  const loom::fabric::FinalizedFabricRoot large =
      generateSystem(fixture, adg::BuiltinTargetPreset::Large);
  lowConfidencePhysicalBehavior(fixture, small, large);
  calibratedPredictionBehavior(fixture, small, large);
  exactCalibrationKernel();
  return EXIT_SUCCESS;
}
