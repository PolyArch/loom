#include "Evaluation/Models/PredictionCalibration.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/ProductionRegistry.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr CaseSubjectRoleRef kParameterRole(0);
constexpr CaseSubjectRoleRef kEvidenceRole(1);
constexpr unsigned kArithmeticWidth = 1024;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("prediction_calibration_invalid: ") + message);
}

void multiplyPowerOfTen(llvm::APInt &value, std::uint64_t exponent) {
  const llvm::APInt ten(value.getBitWidth(), 10);
  for (std::uint64_t index = 0; index != exponent; ++index)
    value *= ten;
}

llvm::Expected<DecimalValue> symmetricRelativeError(DecimalValue predicted,
                                                    DecimalValue observed) {
  if (predicted.coefficient() < 0 || observed.coefficient() < 0)
    return invalid("calibration metrics must be nonnegative");
  if (predicted == observed)
    return DecimalValue::get(0, 0);
  if (predicted.coefficient() == 0 || observed.coefficient() == 0)
    return DecimalValue::get(2, 0);

  const std::int64_t lowExponent =
      std::min(predicted.base10Exponent(), observed.base10Exponent());
  const std::int64_t predictedShift = predicted.base10Exponent() - lowExponent;
  const std::int64_t observedShift = observed.base10Exponent() - lowExponent;
  if (predictedShift > 64 || observedShift > 64)
    return DecimalValue::get(2, 0);

  llvm::APInt predictedInteger(
      kArithmeticWidth, static_cast<std::uint64_t>(predicted.coefficient()));
  llvm::APInt observedInteger(
      kArithmeticWidth, static_cast<std::uint64_t>(observed.coefficient()));
  multiplyPowerOfTen(predictedInteger,
                     static_cast<std::uint64_t>(predictedShift));
  multiplyPowerOfTen(observedInteger,
                     static_cast<std::uint64_t>(observedShift));

  llvm::APInt difference = predictedInteger.uge(observedInteger)
                               ? predictedInteger - observedInteger
                               : observedInteger - predictedInteger;
  llvm::APInt numerator = difference.shl(1);
  llvm::APInt denominator = predictedInteger + observedInteger;

  std::int64_t order = 0;
  llvm::APInt normalized = numerator;
  if (normalized.ult(denominator)) {
    while (normalized.ult(denominator)) {
      normalized *= llvm::APInt(kArithmeticWidth, 10);
      --order;
    }
  } else {
    llvm::APInt next = denominator * llvm::APInt(kArithmeticWidth, 10);
    while (normalized.uge(next)) {
      denominator = std::move(next);
      ++order;
      next = denominator * llvm::APInt(kArithmeticWidth, 10);
    }
    denominator = predictedInteger + observedInteger;
  }

  const std::int64_t resultExponent = order - 17;
  llvm::APInt scaledNumerator = numerator;
  llvm::APInt scaledDenominator = denominator;
  if (resultExponent < 0)
    multiplyPowerOfTen(scaledNumerator,
                       static_cast<std::uint64_t>(-resultExponent));
  else
    multiplyPowerOfTen(scaledDenominator,
                       static_cast<std::uint64_t>(resultExponent));

  llvm::APInt quotient = scaledNumerator.udiv(scaledDenominator);
  llvm::APInt remainder = scaledNumerator.urem(scaledDenominator);
  const llvm::APInt twiceRemainder = remainder.shl(1);
  if (twiceRemainder.ugt(scaledDenominator) ||
      (twiceRemainder == scaledDenominator && quotient[0]))
    ++quotient;
  if (quotient.getActiveBits() > 63)
    return invalid("finalized calibration error exceeds int64");
  return DecimalValue::get(static_cast<std::int64_t>(quotient.getZExtValue()),
                           resultExponent);
}

llvm::Expected<ExactRatio> quantile(const MetricRequest &request) {
  std::optional<ExactRatio> selected;
  for (const EvaluationCondition &condition : request.conditions()) {
    if (condition.kind() != EvaluationConditionKind::Quantile)
      continue;
    if (selected)
      return invalid("metric request contains duplicate Quantile conditions");
    selected = std::get<QuantileCondition>(condition.payload).probability;
  }
  if (!selected)
    return invalid("metric request omits its required Quantile condition");
  return *selected;
}

llvm::Expected<DecimalValue> nearestRank(llvm::ArrayRef<DecimalValue> values,
                                         ExactRatio probability) {
  if (values.empty())
    return invalid("calibration sample set is empty");
  std::vector<DecimalValue> sorted(values.begin(), values.end());
  llvm::sort(sorted, [](DecimalValue lhs, DecimalValue rhs) {
    return compareDecimalValue(lhs, rhs) < 0;
  });
  const unsigned __int128 product =
      static_cast<unsigned __int128>(probability.numerator()) * sorted.size();
  const unsigned __int128 rank =
      (product + probability.denominator() - 1) / probability.denominator();
  std::size_t index = rank == 0 ? 0 : static_cast<std::size_t>(rank - 1);
  index = std::min(index, sorted.size() - 1);
  return sorted[index];
}

llvm::Expected<FinalizedModelParameterBundle>
parameterBundle(const EvaluationRequest &request,
                const ArtifactStore &artifactStore,
                const BlobStore &blobStore) {
  const auto parameters = request.subjectBindings().subjects(kParameterRole);
  if (parameters.size() != 1)
    return invalid("model parameter subject is not total");
  return importModelParameterBundle(parameters.front(), artifactStore,
                                    blobStore);
}

llvm::Expected<EvaluationModelResult>
evaluateFpa(const EvaluationRequest &request,
            const CaseArtifactResolution &resolution,
            const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto bundle = parameterBundle(request, artifactStore, blobStore);
  if (!bundle)
    return bundle.takeError();
  const auto *parameters = bundle->parametersIf<FpaGbdtParameters>();
  if (!parameters)
    return invalid("FPA calibration received a foreign parameter payload");
  const auto evidence = request.subjectBindings().subjects(kEvidenceRole);
  std::array<std::vector<DecimalValue>, 4> errors;
  for (const ArtifactRootReference &reference : evidence) {
    auto sample = importFpaTrainingEvidenceSample(reference, resolution,
                                                  artifactStore, blobStore);
    if (!sample)
      return sample.takeError();
    if (llvm::ArrayRef<std::uint8_t>(sample->groundTruthTargetKey) !=
        parameters->groundTruthTargetKey())
      return invalid("FPA calibration target key changed after case admission");
    auto inference = inferFpaGbdtParameters(*parameters, sample->features);
    if (!inference)
      return inference.takeError();
    if (std::holds_alternative<OutOfDomainModelParameterInference>(*inference))
      return EvaluationModelResult{
          {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    const auto *prediction = std::get<ModelParameterPrediction>(*inference)
                                 .view.getIf<FpaMetricPredictionView>();
    if (!prediction)
      return invalid("FPA contract returned a foreign prediction view");
    const std::array predicted = {
        prediction->limitingClockFrequency, prediction->totalArea,
        prediction->dynamicPower, prediction->leakagePower};
    const std::array observed = {sample->observation.limitingClockFrequency,
                                 sample->observation.totalArea,
                                 sample->observation.dynamicPower,
                                 sample->observation.leakagePower};
    for (std::size_t head = 0; head != errors.size(); ++head) {
      auto error = symmetricRelativeError(predicted[head], observed[head]);
      if (!error)
        return error.takeError();
      errors[head].push_back(*error);
    }
  }

  std::vector<MetricResult> results;
  results.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    std::size_t head = errors.size();
    switch (metric.query().metric) {
    case MetricKind::LimitingClockFrequencyPredictionError:
      head = 0;
      break;
    case MetricKind::TotalAreaPredictionError:
      head = 1;
      break;
    case MetricKind::DynamicPowerPredictionError:
      head = 2;
      break;
    case MetricKind::LeakagePowerPredictionError:
      head = 3;
      break;
    default:
      return invalid("FPA calibration request contains a foreign metric");
    }
    auto probability = quantile(metric);
    if (!probability)
      return probability.takeError();
    auto value = nearestRank(errors[head], *probability);
    if (!value)
      return value.takeError();
    results.push_back({UncertaintyKind::Unquantified,
                       PointObservation{MetricValue{*value}},
                       {}});
  }
  return EvaluationModelResult{{}, CompletedEvidence{std::move(results), {}}};
}

llvm::Expected<EvaluationModelResult> evaluateRuntime(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto bundle = parameterBundle(request, artifactStore, blobStore);
  if (!bundle)
    return bundle.takeError();
  const auto *parameters = bundle->parametersIf<SystemRuntimeGbdtParameters>();
  if (!parameters)
    return invalid(
        "System Runtime calibration received a foreign parameter payload");
  std::vector<DecimalValue> errors;
  for (const ArtifactRootReference &reference :
       request.subjectBindings().subjects(kEvidenceRole)) {
    auto sample = importSystemRuntimeTrainingEvidenceSample(
        reference, resolution, artifactStore, blobStore);
    if (!sample)
      return sample.takeError();
    if (llvm::ArrayRef<std::uint8_t>(sample->groundTruthTargetKey) !=
        parameters->groundTruthTargetKey())
      return invalid("System Runtime target key changed after case admission");
    auto inference =
        inferSystemRuntimeGbdtParameters(*parameters, sample->features);
    if (!inference)
      return inference.takeError();
    if (std::holds_alternative<OutOfDomainModelParameterInference>(*inference))
      return EvaluationModelResult{
          {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    const auto *prediction = std::get<ModelParameterPrediction>(*inference)
                                 .view.getIf<SystemRuntimePredictionView>();
    if (!prediction)
      return invalid(
          "System Runtime contract returned a foreign prediction view");
    auto error = symmetricRelativeError(prediction->runtime, sample->runtime);
    if (!error)
      return error.takeError();
    errors.push_back(*error);
  }
  if (request.metricRequests().size() != 1 ||
      request.metricRequests().front().query().metric !=
          MetricKind::RuntimePredictionError)
    return invalid("System Runtime calibration metric shape is invalid");
  auto probability = quantile(request.metricRequests().front());
  if (!probability)
    return probability.takeError();
  auto value = nearestRank(errors, *probability);
  if (!value)
    return value.takeError();
  MetricResult result{
      UncertaintyKind::Unquantified, PointObservation{MetricValue{*value}}, {}};
  return EvaluationModelResult{{}, CompletedEvidence{{std::move(result)}, {}}};
}

EvaluationModelDescriptorRef modelRef(BuiltinEvaluationModel model) {
  return llvm::cantFail(builtinEvaluationModelDescriptorRef(model));
}

const EvaluationModelProvider kProviders[] = {
    {modelRef(BuiltinEvaluationModel::FpaModelParameterCalibration),
     EvaluationModelInProcessProvider{&evaluateFpa}},
    {modelRef(BuiltinEvaluationModel::SystemRuntimeModelParameterCalibration),
     EvaluationModelInProcessProvider{&evaluateRuntime}},
};

} // namespace

llvm::Error registerPredictionCalibrationProviders() {
  for (const EvaluationModelProvider &provider : kProviders)
    if (llvm::Error error = registerEvaluationModelProvider(provider))
      return error;
  return llvm::Error::success();
}

llvm::Expected<DecimalValue>
calculateSymmetricRelativePredictionError(DecimalValue predicted,
                                          DecimalValue observed) {
  return symmetricRelativeError(predicted, observed);
}

llvm::Expected<DecimalValue>
selectNearestRankPredictionError(llvm::ArrayRef<DecimalValue> values,
                                 ExactRatio probability) {
  return nearestRank(values, probability);
}

} // namespace loom::evaluation::models
