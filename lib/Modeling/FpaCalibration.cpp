#include "Evaluation/Models/CalibratedFpa.h"

#include "Config/ResolvedConfig.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::FpaModelParameterCalibration;
constexpr BuiltinEvaluationModel kModel =
    BuiltinEvaluationModel::FpaModelParameterCalibration;
constexpr CaseSubjectRoleRef kBundleRole(0);
constexpr CaseSubjectRoleRef kEvidenceRole(1);
constexpr ScopeFormRef kWholeCaseScope(0);

struct EmptyFpaCalibrationConfig final {};

struct ExactError final {
  llvm::APInt numerator;
  llvm::APInt denominator;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fpa_calibration_invalid: " + message);
}

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

llvm::Error verifyBundleSubject(const ArtifactRootReference &subject,
                                const EvaluationCase &,
                                const EvaluationSubjectBindings &,
                                const CaseArtifactResolution &,
                                const ArtifactStore &artifactStore,
                                const BlobStore &blobStore) {
  auto bundle = importModelParameterBundle(subject, artifactStore, blobStore);
  if (!bundle)
    return bundle.takeError();
  if (bundle->bundle().parameterContract() != fpaModelParameterContractRef())
    return invalid("bundle subject has a foreign parameter contract");
  return llvm::Error::success();
}

llvm::Error verifyEvidenceSubject(const ArtifactRootReference &subject,
                                  const EvaluationCase &,
                                  const EvaluationSubjectBindings &,
                                  const CaseArtifactResolution &,
                                  const ArtifactStore &artifactStore,
                                  const BlobStore &blobStore) {
  auto sample =
      importFpaTrainingEvidenceSample(subject, artifactStore, blobStore);
  if (!sample)
    return sample.takeError();
  return llvm::Error::success();
}

const ArtifactSchemaDescriptor *const kBundleSchemas[] = {
    &modelParameterBundleSchema};
const ArtifactSchemaDescriptor *const kEvidenceSchemas[] = {
    &EvaluationEvidence::artifactSchema};
const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kBundleRole, "fpa_parameter_bundle", SubjectRoleCardinality::ExactlyOne,
     kBundleSchemas, &verifyBundleSubject},
    {kEvidenceRole, "ground_truth_evidence", SubjectRoleCardinality::OneOrMore,
     kEvidenceSchemas, &verifyEvidenceSubject},
};

const EvaluationCaseSignatureDescriptor kCaseSignature{
    builtinEvaluationCaseKind(kCase),
    "fpa_model_parameter_calibration",
    "One exact FPA parameter bundle calibrated against a nonempty exact "
    "ground-truth Evidence set.",
    kSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {},
};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral schema =
      "loom.fpa.calibration.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(schema.data()), schema.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyFpaCalibrationConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyFpaCalibrationConfig>())
    return invalid("config has a foreign owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return invalid("fixed calibration config is not empty");
  return OwnerValue::get(EmptyFpaCalibrationConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

llvm::ArrayRef<ModelConditionCapability> conditionCapabilities() {
  static const std::array<ModelConditionCapability, 1> capabilities = {{
      {metricDescriptor(MetricKind::LimitingClockFrequencyPredictionError)
           .requiredRequestConditionPatterns.front(),
       ConditionDisposition::Required},
  }};
  return capabilities;
}

const ScopeFormRef kWholeCaseScopes[] = {kWholeCaseScope};
constexpr std::uint8_t kPoint = observationFormMask(ObservationForm::Point);
const MetricCapability kMetricCapabilities[] = {
    {MetricKind::LimitingClockFrequencyPredictionError, kWholeCaseScopes,
     kPoint},
    {MetricKind::TotalAreaPredictionError, kWholeCaseScopes, kPoint},
    {MetricKind::DynamicPowerPredictionError, kWholeCaseScopes, kPoint},
    {MetricKind::LeakagePowerPredictionError, kWholeCaseScopes, kPoint},
};

const EvaluationModelDescriptor &descriptor() {
  static const EvaluationModelDescriptor value{
      builtinEvaluationModelKind(kModel),
      "fpa_model_parameter_calibration",
      "loom.fpa.calibration.v1",
      caseSignatureRef(),
      conditionCapabilities(),
      kMetricCapabilities,
      {},
      {},
      {},
      kConfigView,
      {},
      EvaluationExecutionMethod::Analytic,
      {},
      DeterminismContract::Deterministic,
      {},
      ProviderForm::InProcess};
  return value;
}

llvm::Expected<unsigned> exactWidth(DecimalValue lhs, DecimalValue rhs) {
  const __int128 difference =
      lhs.base10Exponent() >= rhs.base10Exponent()
          ? static_cast<__int128>(lhs.base10Exponent()) - rhs.base10Exponent()
          : static_cast<__int128>(rhs.base10Exponent()) - lhs.base10Exponent();
  const __int128 maximum =
      (static_cast<__int128>(std::numeric_limits<unsigned>::max()) - 128) / 4;
  if (difference > maximum)
    return invalid(
        "decimal exponent exceeds the exact arithmetic address space");
  return static_cast<unsigned>(difference * 4 + 128);
}

llvm::APInt powerOfTen(unsigned exponent, unsigned width) {
  llvm::APInt result(width, 1);
  llvm::APInt base(width, 10);
  while (exponent != 0) {
    if ((exponent & 1U) != 0)
      result *= base;
    exponent >>= 1;
    if (exponent != 0)
      base *= base;
  }
  return result;
}

llvm::APInt scaledCoefficient(DecimalValue value, std::int64_t commonExponent,
                              unsigned width) {
  llvm::APInt coefficient(
      width, static_cast<std::uint64_t>(value.coefficient()), false);
  const unsigned shift = static_cast<unsigned>(
      static_cast<__int128>(value.base10Exponent()) - commonExponent);
  if (shift != 0)
    coefficient *= powerOfTen(shift, width);
  return coefficient;
}

llvm::APInt greatestCommonDivisor(llvm::APInt lhs, llvm::APInt rhs) {
  while (!rhs.isZero()) {
    llvm::APInt remainder = lhs.urem(rhs);
    lhs = std::move(rhs);
    rhs = std::move(remainder);
  }
  return lhs;
}

llvm::Expected<ExactError> symmetricRelativeError(DecimalValue predicted,
                                                  DecimalValue observed) {
  if (predicted.coefficient() < 0 || observed.coefficient() < 0)
    return invalid("physical observations must be nonnegative");
  if (predicted.coefficient() == 0 && observed.coefficient() == 0)
    return ExactError{llvm::APInt(1, 0), llvm::APInt(1, 1)};
  auto width = exactWidth(predicted, observed);
  if (!width)
    return width.takeError();
  const std::int64_t commonExponent =
      std::min(predicted.base10Exponent(), observed.base10Exponent());
  llvm::APInt predictedInteger =
      scaledCoefficient(predicted, commonExponent, *width);
  llvm::APInt observedInteger =
      scaledCoefficient(observed, commonExponent, *width);
  llvm::APInt difference = predictedInteger.uge(observedInteger)
                               ? predictedInteger - observedInteger
                               : observedInteger - predictedInteger;
  llvm::APInt numerator = difference.shl(1);
  llvm::APInt denominator = predictedInteger + observedInteger;
  llvm::APInt divisor = greatestCommonDivisor(numerator, denominator);
  if (!divisor.isOne()) {
    numerator = numerator.udiv(divisor);
    denominator = denominator.udiv(divisor);
  }
  return ExactError{std::move(numerator), std::move(denominator)};
}

int compareError(const ExactError &lhs, const ExactError &rhs) {
  const unsigned width =
      std::max(lhs.numerator.getActiveBits() + rhs.denominator.getActiveBits(),
               rhs.numerator.getActiveBits() +
                   lhs.denominator.getActiveBits()) +
      1;
  const llvm::APInt left =
      lhs.numerator.zextOrTrunc(width) * rhs.denominator.zextOrTrunc(width);
  const llvm::APInt right =
      rhs.numerator.zextOrTrunc(width) * lhs.denominator.zextOrTrunc(width);
  return left == right ? 0 : (left.ult(right) ? -1 : 1);
}

llvm::Expected<DecimalValue> finalizeError(const ExactError &error) {
  constexpr std::uint64_t scale = 1000000000000000000ULL;
  const unsigned width = error.numerator.getActiveBits() + 64;
  llvm::APInt scaled = error.numerator.zextOrTrunc(width);
  scaled *= llvm::APInt(width, scale);
  const llvm::APInt denominator = error.denominator.zextOrTrunc(width);
  llvm::APInt quotient = scaled.udiv(denominator);
  const llvm::APInt remainder = scaled.urem(denominator);
  const llvm::APInt doubledRemainder = remainder.zext(width + 1).shl(1);
  const llvm::APInt extendedDenominator = denominator.zext(width + 1);
  if (doubledRemainder.ugt(extendedDenominator) ||
      (doubledRemainder == extendedDenominator && quotient[0]))
    ++quotient;
  if (quotient.getActiveBits() > 63)
    return invalid("final calibration error exceeds DecimalValue");
  return DecimalValue::get(static_cast<std::int64_t>(quotient.getZExtValue()),
                           -18);
}

llvm::Expected<std::size_t> predictionOrdinal(MetricKind metric) {
  switch (metric) {
  case MetricKind::LimitingClockFrequencyPredictionError:
    return 0;
  case MetricKind::TotalAreaPredictionError:
    return 1;
  case MetricKind::DynamicPowerPredictionError:
    return 2;
  case MetricKind::LeakagePowerPredictionError:
    return 3;
  default:
    return invalid("request contains a metric outside FPA calibration");
  }
}

DecimalValue predictionAt(const FpaMetricPredictionView &view,
                          std::size_t ordinal) {
  switch (ordinal) {
  case 0:
    return view.limitingClockFrequency;
  case 1:
    return view.totalArea;
  case 2:
    return view.dynamicPower;
  case 3:
    return view.leakagePower;
  }
  llvm_unreachable("invalid FPA prediction ordinal");
}

llvm::Expected<ExactRatio> requestedQuantile(const MetricRequest &metric) {
  if (metric.conditions().size() != 1)
    return invalid("calibration metric does not carry exactly one Quantile");
  const auto *quantile =
      std::get_if<QuantileCondition>(&metric.conditions().front().payload);
  if (!quantile)
    return invalid("calibration metric condition is not Quantile");
  return quantile->probability;
}

llvm::Expected<std::size_t> nearestRank(ExactRatio quantile,
                                        std::size_t sampleCount) {
  if (sampleCount == 0)
    return invalid("calibration sample set is empty");
  if (quantile.numerator() == 0)
    return 0;
  using u128 = unsigned __int128;
  const u128 product = static_cast<u128>(quantile.numerator()) * sampleCount;
  const u128 rank = product / quantile.denominator() +
                    (product % quantile.denominator() != 0 ? 1 : 0);
  if (rank == 0 || rank > sampleCount)
    return invalid("Quantile lies outside the admitted probability domain");
  return static_cast<std::size_t>(rank - 1);
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request,
         const CaseArtifactResolution &resolution,
         const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  if (request.modelBinding().descriptorRef() != descriptor().reference())
    return invalid("request selects a foreign model descriptor");
  const auto bundles = request.subjectBindings().subjects(kBundleRole);
  const auto evidence = request.subjectBindings().subjects(kEvidenceRole);
  if (bundles.size() != 1 || evidence.empty())
    return invalid("calibration subjects are not total");
  auto bundle =
      importModelParameterBundle(bundles.front(), artifactStore, blobStore);
  if (!bundle)
    return bundle.takeError();
  const auto *parameters = bundle->parametersIf<FpaGbdtParameters>();
  if (!parameters)
    return invalid("bundle payload has a foreign parameter owner type");

  std::array<std::vector<ExactError>, 4> errors;
  for (const ArtifactRootReference &evidenceReference : evidence) {
    auto sample = importFpaTrainingEvidenceSample(evidenceReference,
                                                  artifactStore, blobStore);
    if (!sample)
      return sample.takeError();
    if (llvm::ArrayRef<std::uint8_t>(sample->groundTruthTargetKey) !=
        parameters->groundTruthTargetKey())
      return invalid("ground-truth Evidence target differs from the bundle");
    auto inference =
        inferModelParameters(*bundle, OwnerValue::get(sample->features));
    if (!inference)
      return inference.takeError();
    const auto *prediction = std::get_if<ModelParameterPrediction>(&*inference);
    if (!prediction)
      return EvaluationModelResult{
          {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    const auto *view = prediction->view.getIf<FpaMetricPredictionView>();
    if (!view)
      return invalid("parameter contract returned a foreign prediction view");
    for (std::size_t ordinal = 0; ordinal != errors.size(); ++ordinal) {
      auto error =
          symmetricRelativeError(predictionAt(*view, ordinal),
                                 predictionAt(sample->observation, ordinal));
      if (!error)
        return error.takeError();
      errors[ordinal].push_back(std::move(*error));
    }
  }
  for (auto &metricErrors : errors)
    llvm::sort(metricErrors, [](const ExactError &lhs, const ExactError &rhs) {
      return compareError(lhs, rhs) < 0;
    });

  std::vector<MetricResult> results;
  results.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    auto ordinal = predictionOrdinal(metric.query().metric);
    if (!ordinal)
      return ordinal.takeError();
    auto quantile = requestedQuantile(metric);
    if (!quantile)
      return quantile.takeError();
    auto rank = nearestRank(*quantile, errors[*ordinal].size());
    if (!rank)
      return rank.takeError();
    auto value = finalizeError(errors[*ordinal][*rank]);
    if (!value)
      return value.takeError();
    results.push_back(
        {UncertaintyKind::ExactWithinModel, PointObservation{*value}, {}});
  }
  return EvaluationModelResult{{}, CompletedEvidence{std::move(results), {}}};
}

const EvaluationModelProvider &provider() {
  static const EvaluationModelProvider value{
      descriptor().reference(), EvaluationModelInProcessProvider{&evaluate}};
  return value;
}

} // namespace

llvm::Error registerFpaModelParameterCalibrationModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(descriptor()))
    return error;
  return registerEvaluationModelProvider(provider());
}

llvm::Expected<DecimalValue> calculateFpaPredictionErrorQuantile(
    llvm::ArrayRef<std::pair<DecimalValue, DecimalValue>> samples,
    ExactRatio quantile) {
  if (samples.empty())
    return invalid("calibration sample set is empty");
  std::vector<ExactError> errors;
  errors.reserve(samples.size());
  for (const auto &[predicted, observed] : samples) {
    auto error = symmetricRelativeError(predicted, observed);
    if (!error)
      return error.takeError();
    errors.push_back(std::move(*error));
  }
  llvm::sort(errors, [](const ExactError &lhs, const ExactError &rhs) {
    return compareError(lhs, rhs) < 0;
  });
  auto rank = nearestRank(quantile, errors.size());
  if (!rank)
    return rank.takeError();
  return finalizeError(errors[*rank]);
}

EvaluationCaseSignatureRef fpaModelParameterCalibrationCaseSignatureRef() {
  return caseSignatureRef();
}

EvaluationModelDescriptorRef fpaModelParameterCalibrationModelDescriptorRef() {
  return descriptor().reference();
}

CaseSubjectRoleRef fpaModelParameterCalibrationBundleRole() {
  return kBundleRole;
}

CaseSubjectRoleRef fpaModelParameterCalibrationEvidenceRole() {
  return kEvidenceRole;
}

} // namespace loom::evaluation::models
