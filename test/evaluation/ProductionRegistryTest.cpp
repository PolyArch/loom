#include "Evaluation/ProductionRegistry.h"

#include "Evaluation/Metric.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/PredictionCalibration.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace {

using namespace loom;
using namespace loom::evaluation;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  std::cerr << test.str() << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted a reference outside its exact registry version");
  llvm::consumeError(value.takeError());
}

void registryMajorMatchesCurrentArtifactContracts() {
  const llvm::StringRef test = __func__;
  require(test, evaluationSchemaVersion() == SchemaVersion{3, 0},
          "current Evaluation registry is not 3.0");

  const auto canonicalDataflowFabricCase = builtinEvaluationCaseSignatureRef(
      BuiltinEvaluationCase::CanonicalDataflowWithFabric);
  const auto canonicalDataflowFabricModel = take(
      test, builtinEvaluationModelDescriptorRef(
                BuiltinEvaluationModel::CanonicalDataflowFabricLowConfidence));
  const auto currentCase =
      take(test, EvaluationCaseSignatureRef::get(
                     {3, 0}, canonicalDataflowFabricCase.caseKind()));
  require(test, currentCase.descriptor() &&
                    currentCase.descriptor()->registryVersion ==
                        SchemaVersion{3, 0},
          "production case kind 1 is not registered in registry 3.0");

  const auto currentModel =
      take(test, EvaluationModelDescriptorRef::get(
                     {3, 0}, canonicalDataflowFabricModel.modelKind()));
  require(test,
          currentModel.descriptor() &&
              currentModel.descriptor()->registryVersion ==
                  SchemaVersion{3, 0} &&
              currentModel.descriptor()->caseSignature.schemaVersion() ==
                  SchemaVersion{3, 0},
          "production model kind 3 is not registered in registry 3.0");

  for (SchemaVersion obsolete :
       {SchemaVersion{2, 0}, SchemaVersion{2, 1}}) {
    expectError(test, EvaluationCaseSignatureRef::get(
                          obsolete, canonicalDataflowFabricCase.caseKind()));
    expectError(test, EvaluationModelDescriptorRef::get(
                          obsolete, canonicalDataflowFabricModel.modelKind()));
  }
}

void appendedMetricsAndModelSlotsMatchTheCatalog() {
  const llvm::StringRef test = __func__;
  struct PredictionErrorContract final {
    MetricKind metric;
    BuiltinEvaluationCase calibrationCase;
  };
  const std::array predictionErrors = {
      PredictionErrorContract{
          MetricKind::LimitingClockFrequencyPredictionError,
          BuiltinEvaluationCase::FpaModelParameterCalibration},
      PredictionErrorContract{
          MetricKind::TotalAreaPredictionError,
          BuiltinEvaluationCase::FpaModelParameterCalibration},
      PredictionErrorContract{
          MetricKind::DynamicPowerPredictionError,
          BuiltinEvaluationCase::FpaModelParameterCalibration},
      PredictionErrorContract{
          MetricKind::LeakagePowerPredictionError,
          BuiltinEvaluationCase::FpaModelParameterCalibration},
      PredictionErrorContract{
          MetricKind::RuntimePredictionError,
          BuiltinEvaluationCase::SystemRuntimeModelParameterCalibration}};
  for (const PredictionErrorContract &contract : predictionErrors) {
    const MetricDescriptor &descriptor = metricDescriptor(contract.metric);
    const auto *domain =
        std::get_if<ClosedDecimalIntervalMetricDomain>(&descriptor.valueDomain);
    require(test,
            descriptor.dimension == MetricDimension::Dimensionless && domain &&
                domain->lower.coefficient() == 0 &&
                domain->upper.coefficient() == 2 &&
                domain->upper.base10Exponent() == 0,
            "prediction-error metric does not own the exact [0,2] domain");
    require(test,
            descriptor.permittedRequestConditionPatterns.size() == 1 &&
                descriptor.requiredRequestConditionPatterns.size() == 1 &&
                descriptor.permittedRequestConditionPatterns.front() ==
                    descriptor.requiredRequestConditionPatterns.front() &&
                descriptor.requiredRequestConditionPatterns.front().kind ==
                    EvaluationConditionKind::Quantile &&
                descriptor.requiredRequestConditionPatterns.front()
                        .targets.caseSignature.caseKind() ==
                    builtinEvaluationCaseKind(contract.calibrationCase),
            "prediction-error metric does not require its exact Quantile");
  }

  const auto lowConfidenceFabricRef =
      take(test, builtinEvaluationModelDescriptorRef(
                     BuiltinEvaluationModel::FabricLowConfidence));
  const auto calibratedFabricRef =
      take(test, builtinEvaluationModelDescriptorRef(
                     BuiltinEvaluationModel::FabricCalibratedFpa));
  const auto runtimeCalibrationRef =
      take(test,
           builtinEvaluationModelDescriptorRef(
               BuiltinEvaluationModel::SystemRuntimeModelParameterCalibration));
  const auto systemRuntimePredictorRef =
      take(test, builtinEvaluationModelDescriptorRef(
                     BuiltinEvaluationModel::Gem5CgraSystemRuntimePredictor));
  const EvaluationModelDescriptor *lowConfidenceFabricModel =
      lowConfidenceFabricRef.descriptor();
  const EvaluationModelDescriptor *calibratedFabricModel =
      calibratedFabricRef.descriptor();
  const EvaluationModelDescriptor *runtimeCalibrationModel =
      runtimeCalibrationRef.descriptor();
  const EvaluationModelDescriptor *systemRuntimePredictor =
      systemRuntimePredictorRef.descriptor();
  require(
      test,
      lowConfidenceFabricModel->caseSignature.caseKind() ==
              builtinEvaluationCaseKind(
                  BuiltinEvaluationCase::FabricHardwareAnalysis) &&
          lowConfidenceFabricModel->inputSlots.empty() &&
          calibratedFabricModel->inputSlots.size() == 1 &&
          calibratedFabricModel->inputSlots.front().modelParameterContract &&
          runtimeCalibrationModel->inputSlots.empty() &&
          systemRuntimePredictor->inputSlots.front().modelParameterContract ==
              models::systemRuntimeModelParameterContractRef(),
      "production parameter slots do not reference their exact contracts");
  require(test,
          calibratedFabricModel->inputSlots.front().modelParameterContract !=
              systemRuntimePredictor->inputSlots.front().modelParameterContract,
          "FPA and System Runtime parameter contracts were conflated");

  const std::array parameterBackedFpaModels = {
      BuiltinEvaluationModel::StructuredFabricCalibratedFpa,
      BuiltinEvaluationModel::CanonicalDataflowFabricCalibratedFpa,
      BuiltinEvaluationModel::FabricCalibratedFpa};
  for (BuiltinEvaluationModel model : parameterBackedFpaModels) {
    const auto reference =
        take(test, builtinEvaluationModelDescriptorRef(model));
    require(
        test,
        reference.descriptor()->providerForm == ProviderForm::InProcess &&
            reference.descriptor()->inputSlots.size() == 1 &&
            reference.descriptor()->inputSlots.front().modelParameterContract ==
                models::fpaModelParameterContractRef(),
        "parameter-backed FPA descriptor is incomplete");
  }
  const auto fpaCalibration =
      take(test, builtinEvaluationModelDescriptorRef(
                     BuiltinEvaluationModel::FpaModelParameterCalibration));
  require(test, fpaCalibration.descriptor()->inputSlots.empty(),
          "FPA calibration duplicated its candidate subject as a model input");

  const std::array gem5ExecutionModels = {
      BuiltinEvaluationModel::Gem5SystemDfg,
      BuiltinEvaluationModel::Gem5SystemCgra,
      BuiltinEvaluationModel::Gem5SystemRtl};
  for (BuiltinEvaluationModel model : gem5ExecutionModels) {
    const auto gem5ExecutionRef =
        take(test, builtinEvaluationModelDescriptorRef(model));
    const EvaluationModelDescriptor *gem5ExecutionModel =
        gem5ExecutionRef.descriptor();
    require(test,
            gem5ExecutionModel->caseSignature.caseKind() ==
                    builtinEvaluationCaseKind(
                        BuiltinEvaluationCase::SystemSimulation) &&
                gem5ExecutionModel->outputSlots.size() == 1 &&
                *gem5ExecutionModel->outputSlots.front().schema ==
                    sim::simulationExecutionSchema,
            "gem5 descriptor does not publish the System execution contract");
  }
}

void calibrationArithmeticIsExactAndDeterministic() {
  const llvm::StringRef test = __func__;
  const auto decimal = [&](std::int64_t coefficient,
                           std::int64_t exponent = 0) {
    return take(test, DecimalValue::get(coefficient, exponent));
  };
  const DecimalValue zero = decimal(0);
  const DecimalValue two = decimal(2);
  require(test,
          take(test, models::calculateSymmetricRelativePredictionError(
                         zero, zero)) == zero,
          "zero prediction and observation did not produce zero error");
  require(test,
          take(test, models::calculateSymmetricRelativePredictionError(
                         decimal(1), zero)) == two,
          "one-sided zero did not produce the maximum symmetric error");
  require(test,
          take(test, models::calculateSymmetricRelativePredictionError(
                         decimal(100), decimal(80))) ==
              decimal(222222222222222222LL, -18),
          "symmetric relative error was not finalized to 18 digits");

  const std::array values = {decimal(3, -1), decimal(1, -1), decimal(2, -1)};
  const ExactRatio median = take(test, ExactRatio::get(1, 2));
  require(test,
          take(test, models::selectNearestRankPredictionError(
                         values, median)) == decimal(2, -1),
          "nearest-rank median selected the wrong sample");
}

} // namespace

int main() {
  requireSuccess("registration", registerProductionEvaluationRegistry());
  registryMajorMatchesCurrentArtifactContracts();
  appendedMetricsAndModelSlotsMatchTheCatalog();
  calibrationArithmeticIsExactAndDeterministic();
  return EXIT_SUCCESS;
}
