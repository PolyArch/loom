#include "Evaluation/ProductionRegistry.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Metric.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "ImplementationPlatform/ImplementationPlatform.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <system_error>
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

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) {
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-registry", path_))
      fail(test, error.message());
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      std::cerr << "temporary directory cleanup failed: " << error.message()
                << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

void exactRegistryVersionsRemainDistinct() {
  const llvm::StringRef test = __func__;
  require(test, evaluationSchemaVersion() == SchemaVersion{2, 1},
          "current Evaluation registry is not 2.1");

  const auto canonicalDataflowFabricCase = builtinEvaluationCaseSignatureRef(
      BuiltinEvaluationCase::CanonicalDataflowWithFabric);
  const auto canonicalDataflowFabricModel = take(
      test, builtinEvaluationModelDescriptorRef(
                BuiltinEvaluationModel::CanonicalDataflowFabricLowConfidence));
  const auto currentCase =
      take(test, EvaluationCaseSignatureRef::get(
                     {2, 1}, canonicalDataflowFabricCase.caseKind()));
  const auto legacyCase =
      take(test, EvaluationCaseSignatureRef::get(
                     {2, 0}, canonicalDataflowFabricCase.caseKind()));
  require(test, currentCase.descriptor() && legacyCase.descriptor(),
          "production case kind 1 was not registered in both catalogs");
  require(test,
          currentCase.descriptor() != legacyCase.descriptor() &&
              currentCase.descriptor()->registryVersion ==
                  SchemaVersion{2, 1} &&
              legacyCase.descriptor()->registryVersion == SchemaVersion{2, 0},
          "2.0 case reference aliases the 2.1 descriptor view");

  const auto currentModel =
      take(test, EvaluationModelDescriptorRef::get(
                     {2, 1}, canonicalDataflowFabricModel.modelKind()));
  const auto legacyModel =
      take(test, EvaluationModelDescriptorRef::get(
                     {2, 0}, canonicalDataflowFabricModel.modelKind()));
  require(test, currentModel.descriptor() && legacyModel.descriptor(),
          "production model kind 3 was not registered in both catalogs");
  require(test,
          currentModel.descriptor() != legacyModel.descriptor() &&
              legacyModel.descriptor()->registryVersion ==
                  SchemaVersion{2, 0} &&
              legacyModel.descriptor()->caseSignature.schemaVersion() ==
                  SchemaVersion{2, 0},
          "2.0 model reference aliases or embeds a 2.1 descriptor");

  const std::array appendedCases = {
      BuiltinEvaluationCase::FabricHardwareAnalysis,
      BuiltinEvaluationCase::SystemRuntimeModelParameterCalibration,
      BuiltinEvaluationCase::MappedRtlSimulation};
  for (BuiltinEvaluationCase evaluationCase : appendedCases) {
    const auto reference =
        take(test, EvaluationCaseSignatureRef::get(
                       {2, 1}, builtinEvaluationCaseKind(evaluationCase)));
    require(test, reference.descriptor() != nullptr,
            "2.1 production case descriptor is unresolved");
    expectError(test, EvaluationCaseSignatureRef::get(
                          {2, 0}, builtinEvaluationCaseKind(evaluationCase)));
  }
  const std::array appendedModels = {
      BuiltinEvaluationModel::FabricLowConfidence,
      BuiltinEvaluationModel::FabricCalibratedFpa,
      BuiltinEvaluationModel::SystemRuntimeModelParameterCalibration,
      BuiltinEvaluationModel::Gem5CgraSystemRuntimePredictor,
      BuiltinEvaluationModel::Gem5SystemDfg,
      BuiltinEvaluationModel::Gem5SystemCgra,
      BuiltinEvaluationModel::Gem5SystemRtl,
      BuiltinEvaluationModel::OpenRoadRoutedStaticFpa,
      BuiltinEvaluationModel::MappedRtlSimulator};
  for (BuiltinEvaluationModel model : appendedModels) {
    const auto reference =
        take(test, builtinEvaluationModelDescriptorRef(model));
    require(test, reference.descriptor() != nullptr,
            "2.1 production model descriptor is unresolved");
    expectError(test, EvaluationModelDescriptorRef::get(
                          {2, 0}, builtinEvaluationModelKind(model)));
  }
}

void legacyRequestRoundTripsWithoutVersionReinterpretation() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore artifacts(directory.path());
  BlobStore blobs(directory.path());

  const ArtifactIdentity dataflowIdentity =
      take(test, artifacts.put(dataflow::canonicalDataflowSchema,
                               CanonicalSemanticBytes({0x11})));
  const ArtifactIdentity fabricIdentity =
      take(test, artifacts.put(loom::fabric::fabricArtifactSchema,
                               CanonicalSemanticBytes({0x22})));
  const ArtifactRootReference dataflow{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, dataflowIdentity};
  const ArtifactRootReference fabricRoot{
      loom::fabric::fabricArtifactSchema.identity.str(),
      loom::fabric::fabricArtifactSchema.version, fabricIdentity};
  CaseArtifactResolution resolution = take(
      test, CaseArtifactResolution::get({{dataflow, {}}, {fabricRoot, {}}}));
  EvaluationSubjectBindings subjects = take(
      test,
      EvaluationSubjectBindings::get({{CaseSubjectRoleRef(0), {dataflow}},
                                      {CaseSubjectRoleRef(1), {fabricRoot}}}));
  const auto currentCase = builtinEvaluationCaseSignatureRef(
      BuiltinEvaluationCase::CanonicalDataflowWithFabric);
  const auto caseRef = take(
      test, EvaluationCaseSignatureRef::get({2, 0}, currentCase.caseKind()));
  EvaluationCase evaluationCase =
      take(test,
           EvaluationCase::get(caseRef, std::move(subjects), std::nullopt,
                               std::nullopt, {}, resolution, artifacts, blobs));
  MetricRequest metric =
      take(test, MetricRequest::get({MetricKind::Runtime,
                                     EvaluationScope{ScopeFormRef(0), {}}},
                                    {}, evaluationCase, resolution, artifacts));
  const auto currentModel = take(
      test, builtinEvaluationModelDescriptorRef(
                BuiltinEvaluationModel::CanonicalDataflowFabricLowConfidence));
  const auto modelRef =
      take(test,
           EvaluationModelDescriptorRef::get({2, 0}, currentModel.modelKind()));
  ResolvedModelBinding binding =
      take(test, ResolvedModelBinding::project(modelRef, {},
                                               defaultResolvedConfig()));
  EvaluationRequest request =
      take(test, EvaluationRequest::get(evaluationCase, {metric}, {},
                                        std::move(binding), 0, resolution,
                                        artifacts, blobs));
  const ArtifactRootReference reference =
      take(test, publishEvaluationRequest(request, artifacts));
  EvaluationRequest imported = take(
      test, importEvaluationRequest(reference, resolution, artifacts, blobs));
  require(test,
          imported.modelBinding().descriptorRef().schemaVersion() ==
                  SchemaVersion{2, 0} &&
              imported.modelBinding()
                      .descriptorRef()
                      .descriptor()
                      ->caseSignature.schemaVersion() == SchemaVersion{2, 0},
          "legacy Request import reinterpreted its registry references");
  require(test,
          EvaluationRequest::artifactSchema.version == SchemaVersion{1, 0},
          "registry extension changed the Request root schema");
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
          calibratedFabricModel->inputSlots.size() == 2 &&
          calibratedFabricModel->inputSlots.front().modelParameterContract &&
          *calibratedFabricModel->inputSlots[1].acceptedSchemas.front() ==
              platform::implementationPlatformSchema &&
          runtimeCalibrationModel->inputSlots.size() == 1 &&
          runtimeCalibrationModel->inputSlots.front().modelParameterContract &&
          systemRuntimePredictor->inputSlots.front().modelParameterContract ==
              runtimeCalibrationModel->inputSlots.front()
                  .modelParameterContract,
      "production parameter slots do not reference their exact contracts");
  require(
      test,
      calibratedFabricModel->inputSlots.front().modelParameterContract !=
          runtimeCalibrationModel->inputSlots.front().modelParameterContract,
      "FPA and System Runtime parameter contracts were conflated");

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

} // namespace

int main() {
  requireSuccess("registration", registerProductionEvaluationRegistry());
  exactRegistryVersionsRemainDistinct();
  legacyRequestRoundTripsWithoutVersionReinterpretation();
  appendedMetricsAndModelSlotsMatchTheCatalog();
  return EXIT_SUCCESS;
}
