#include "Evaluation/ProductionRegistry.h"

#include "Deployment/Deployment.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Models/CalibratedFpa.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/CanonicalDataflowFunctional.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/FabricLowConfidence.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Models/Gem5SystemCgra.h"
#include "Evaluation/Models/Gem5SystemDfg.h"
#include "Evaluation/Models/Gem5SystemRtl.h"
#include "Evaluation/Models/MappedRtlSimulation.h"
#include "Evaluation/Models/OpenRoadStaticFpa.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "Evaluation/Models/PredictionCalibration.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/Models/SystemRuntimePredictor.h"
#include "Evaluation/OwnerError.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "AnalyticModelSupport.h"

#include "Config/ResolvedConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

constexpr BuiltinEvaluationCase kSystemCase =
    BuiltinEvaluationCase::SystemSimulation;
constexpr BuiltinEvaluationCase kHardwareCase =
    BuiltinEvaluationCase::FabricHardwareAnalysis;
constexpr BuiltinEvaluationCase kFpaCalibrationCase =
    BuiltinEvaluationCase::FpaModelParameterCalibration;
constexpr BuiltinEvaluationCase kRuntimeCalibrationCase =
    BuiltinEvaluationCase::SystemRuntimeModelParameterCalibration;
constexpr BuiltinEvaluationCase kMappedRtlCase =
    BuiltinEvaluationCase::MappedRtlSimulation;

constexpr CaseSubjectRoleRef kRole0(0);
constexpr CaseSubjectRoleRef kRole1(1);
constexpr ScopeFormRef kWholeCase(0);
constexpr ModelInputSlotRef kParameterInput(0);
constexpr ModelOutputSlotRef kExecutionOutput(0);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("production_evaluation_registry_invalid: ") + message);
}

llvm::Expected<std::shared_ptr<const deployment::FinalizedDeployment>>
importCachedDeployment(const ArtifactRootReference &reference,
                       const ArtifactStore &artifacts, const BlobStore &blobs);

EvaluationCaseSignatureRef caseRef(BuiltinEvaluationCase evaluationCase) {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(evaluationCase)));
}

const ArtifactSchemaDescriptor *const kFabricSchemas[] = {
    &fabric::fabricArtifactSchema};
const ArtifactSchemaDescriptor *const kDeploymentSchemas[] = {
    &deployment::deploymentSchema};
const ArtifactSchemaDescriptor *const kGem5Schemas[] = {
    &runtime::gem5SimulationBindingSchema};
const ArtifactSchemaDescriptor *const kHardwareSchemas[] = {
    &hardware::hardwareImplementationSchema};
const ArtifactSchemaDescriptor *const kParameterSchemas[] = {
    &modelParameterBundleSchema};
const ArtifactSchemaDescriptor *const kEvidenceSchemas[] = {
    &EvaluationEvidence::artifactSchema};
const ArtifactSchemaDescriptor *const kWorkloadSchemas[] = {
    &sim::simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const kRuntimeInputSchemas[] = {
    &sim::simulationRuntimeInputSchema};

llvm::Error verifyGem5Binding(const ArtifactRootReference &subject,
                              const EvaluationCase &,
                              const EvaluationSubjectBindings &bindings,
                              const CaseArtifactResolution &,
                              const ArtifactStore &artifacts,
                              const BlobStore &blobs) {
  const auto deployments = bindings.subjects(kRole0);
  if (deployments.size() != 1)
    return invalid("gem5 binding compatibility requires one Deployment");
  auto deployment =
      importCachedDeployment(deployments.front(), artifacts, blobs);
  if (!deployment)
    return deployment.takeError();
  auto gem5 = runtime::importGem5SimulationBinding(subject, artifacts);
  if (!gem5)
    return gem5.takeError();
  auto systemMapping = mapping::importSystemMapping(
      (*deployment)->deployment().systemMapping(), artifacts);
  if (!systemMapping)
    return systemMapping.takeError();
  if (systemMapping->view().fabricIdentity() !=
      gem5->binding().fabric().artifact)
    return invalid("gem5 binding names a foreign System Fabric");

  return llvm::Error::success();
}

llvm::Error
verifyParameterSubject(const ArtifactRootReference &subject,
                       const ModelParameterContractRef &expectedContract,
                       const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto bundle = importModelParameterBundle(subject, artifacts, blobs);
  if (!bundle)
    return bundle.takeError();
  if (bundle->bundle().parameterContract() != expectedContract)
    return invalid("model parameter subject has the wrong contract");
  return llvm::Error::success();
}

llvm::Error verifySystemRuntimeParameterSubject(
    const ArtifactRootReference &subject, const EvaluationCase &,
    const EvaluationSubjectBindings &, const CaseArtifactResolution &,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return verifyParameterSubject(
      subject, models::systemRuntimeModelParameterContractRef(), artifacts,
      blobs);
}

llvm::Error verifyFpaParameterSubject(const ArtifactRootReference &subject,
                                      const EvaluationCase &,
                                      const EvaluationSubjectBindings &,
                                      const CaseArtifactResolution &,
                                      const ArtifactStore &artifacts,
                                      const BlobStore &blobs) {
  return verifyParameterSubject(subject, models::fpaModelParameterContractRef(),
                                artifacts, blobs);
}

llvm::Error verifyFpaGroundTruthSubject(
    const ArtifactRootReference &subject, const EvaluationCase &,
    const EvaluationSubjectBindings &bindings,
    const CaseArtifactResolution &resolution, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  const auto parameters = bindings.subjects(kRole0);
  if (parameters.size() != 1)
    return invalid("FPA calibration requires one parameter bundle");
  auto bundle =
      importModelParameterBundle(parameters.front(), artifacts, blobs);
  if (!bundle)
    return bundle.takeError();
  const auto *typed = bundle->parametersIf<models::FpaGbdtParameters>();
  if (!typed)
    return invalid("FPA calibration bundle has a foreign payload");
  auto sample = models::importFpaTrainingEvidenceSample(subject, resolution,
                                                        artifacts, blobs);
  if (!sample)
    return sample.takeError();
  if (llvm::ArrayRef<std::uint8_t>(sample->groundTruthTargetKey) !=
      typed->groundTruthTargetKey())
    return invalid("FPA calibration Evidence has a foreign target key");
  return llvm::Error::success();
}

llvm::Error verifySystemRuntimeGroundTruthSubject(
    const ArtifactRootReference &subject, const EvaluationCase &,
    const EvaluationSubjectBindings &bindings,
    const CaseArtifactResolution &resolution, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  const auto parameters = bindings.subjects(kRole0);
  if (parameters.size() != 1)
    return invalid("System Runtime calibration requires one parameter bundle");
  auto bundle =
      importModelParameterBundle(parameters.front(), artifacts, blobs);
  if (!bundle)
    return bundle.takeError();
  const auto *typed =
      bundle->parametersIf<models::SystemRuntimeGbdtParameters>();
  if (!typed)
    return invalid("System Runtime calibration bundle has a foreign payload");
  auto sample = models::importSystemRuntimeTrainingEvidenceSample(
      subject, resolution, artifacts, blobs);
  if (!sample)
    return sample.takeError();
  if (llvm::ArrayRef<std::uint8_t>(sample->groundTruthTargetKey) !=
      typed->groundTruthTargetKey())
    return invalid(
        "System Runtime calibration Evidence has a foreign target key");
  return llvm::Error::success();
}

bool reaches(const CaseArtifactResolution &resolution,
             const ArtifactRootReference &owner,
             const ArtifactRootReference &dependency) {
  const CaseArtifactResolution::Entry *entry = resolution.find(owner);
  return entry && CaseArtifactResolution::reaches(*entry, dependency);
}

llvm::Expected<std::shared_ptr<const deployment::FinalizedDeployment>>
importCachedDeployment(const ArtifactRootReference &reference,
                       const ArtifactStore &artifacts, const BlobStore &blobs) {
  const std::array<ArtifactRootReference, 1> references{reference};
  return importCachedArtifact<deployment::FinalizedDeployment>(
      artifacts, &blobs, references, [&] {
        return deployment::importDeployment(reference, artifacts, blobs);
      });
}

llvm::Expected<std::shared_ptr<const sim::ImportedSpatialSimulationInputs>>
importCachedSpatialInputs(const ArtifactRootReference &workload,
                          const ArtifactRootReference &runtimeInput,
                          const ArtifactStore &artifacts) {
  const std::array<ArtifactRootReference, 2> references{workload, runtimeInput};
  return importCachedArtifact<sim::ImportedSpatialSimulationInputs>(
      artifacts, nullptr, references, [&] {
        return sim::importSpatialSimulationInputs(workload, runtimeInput,
                                                  artifacts);
      });
}

llvm::Error
verifySystemWorkload(const EvaluationCase &,
                     const EvaluationSubjectBindings &bindings,
                     const std::optional<ArtifactRootReference> &workload,
                     const std::optional<ArtifactRootReference> &runtimeInput,
                     const CaseArtifactResolution &resolution,
                     const ArtifactStore &, const BlobStore &) {
  const auto deployments = bindings.subjects(kRole0);
  const auto gem5 = bindings.subjects(kRole1);
  if (deployments.size() != 1 || gem5.size() != 1 || !workload || !runtimeInput)
    return invalid("system case inputs are not total");
  if (!reaches(resolution, *workload, deployments.front()) ||
      !reaches(resolution, *runtimeInput, *workload) ||
      !reaches(resolution, *runtimeInput, deployments.front()))
    return invalid("System workload/runtime input do not reach Deployment");
  return llvm::Error::success();
}

llvm::Error verifyMappedRtlWorkload(
    const EvaluationCase &evaluationCase,
    const EvaluationSubjectBindings &bindings,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  const auto implementations = bindings.subjects(kRole0);
  const auto deployments = bindings.subjects(kRole1);
  if (implementations.size() != 1 || deployments.size() != 1 || !workload ||
      !runtimeInput)
    return invalid("mapped RTL case inputs are not total");
  if (!reaches(resolution, deployments.front(), implementations.front()))
    return invalid("Deployment does not select the HardwareImplementation");
  if (!reaches(resolution, *runtimeInput, *workload))
    return invalid("Spatial runtime input does not reach its workload");

  auto deployment =
      importCachedDeployment(deployments.front(), artifacts, blobs);
  if (!deployment)
    return deployment.takeError();
  auto inputs = importCachedSpatialInputs(*workload, *runtimeInput, artifacts);
  if (!inputs)
    return inputs.takeError();
  const sim::SpatialSimulationWorkload *spatial = (*inputs)->workload.spatial();
  if (!spatial)
    return invalid("mapped RTL case requires a Spatial workload");
  auto selection = deployment::resolveDeploymentSpatialLaunchSelection(
      **deployment, spatial->launchRef, spatial->denseCoordinates, artifacts,
      blobs);
  if (!selection)
    return selection.takeError();
  if (selection->hardwareImplementation != implementations.front())
    return invalid("Deployment selects a foreign HardwareImplementation");
  if (selection->dataflow.artifact != (*inputs)->dataflow.identity())
    return invalid("Deployment and Spatial workload select different "
                   "Dataflow owners");
  (void)evaluationCase;
  return llvm::Error::success();
}

llvm::Expected<SubjectTargetRef>
resolveMappedRtlCycle(const EvaluationCase &evaluationCase,
                      const CaseArtifactResolution &,
                      const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto implementations =
      evaluationCase.subjectBindings().subjects(kRole0);
  const auto deployments = evaluationCase.subjectBindings().subjects(kRole1);
  if (implementations.size() != 1 || deployments.size() != 1 ||
      !evaluationCase.workload() || !evaluationCase.runtimeInput())
    return invalid("mapped RTL reference cycle requires total case inputs");
  auto deployment =
      importCachedDeployment(deployments.front(), artifacts, blobs);
  if (!deployment)
    return deployment.takeError();
  auto inputs = importCachedSpatialInputs(
      *evaluationCase.workload(), *evaluationCase.runtimeInput(), artifacts);
  if (!inputs)
    return inputs.takeError();
  const sim::SpatialSimulationWorkload *spatial = (*inputs)->workload.spatial();
  if (!spatial)
    return invalid("mapped RTL reference cycle requires a Spatial workload");
  auto selection = deployment::resolveDeploymentSpatialLaunchSelection(
      **deployment, spatial->launchRef, spatial->denseCoordinates, artifacts,
      blobs);
  if (!selection)
    return selection.takeError();
  if (selection->hardwareImplementation != implementations.front() ||
      selection->dataflow.artifact != (*inputs)->dataflow.identity())
    return invalid("mapped RTL reference cycle has foreign selected owners");
  return SubjectTargetRef{kRole1, deployments.front(),
                          selection->spatialMapping};
}

SubjectReferenceType mappingRootType() {
  return SubjectReferenceType{ArtifactRootType{mapping::mappingArtifactSchema}};
}

SubjectTargetPattern fabricTarget() {
  return SubjectTargetPattern{kRole0, SubjectReferenceType{ArtifactRootType{
                                          fabric::fabricArtifactSchema}}};
}

const std::vector<ConditionApplicabilityPattern> &hardwareConditions() {
  static const std::vector<ConditionApplicabilityPattern> patterns = {
      {EvaluationConditionKind::ProcessCorner,
       {caseRef(kHardwareCase), {fabricTarget()}}},
      {EvaluationConditionKind::SupplyVoltage,
       {caseRef(kHardwareCase), {fabricTarget()}}},
      {EvaluationConditionKind::Temperature,
       {caseRef(kHardwareCase), {fabricTarget()}}},
      {EvaluationConditionKind::RequiredClockPeriod,
       {caseRef(kHardwareCase), {fabricTarget()}}},
      {EvaluationConditionKind::RelativeClockSchedule,
       {caseRef(kHardwareCase), {fabricTarget(), fabricTarget()}}},
      {EvaluationConditionKind::ActivityBinding,
       {caseRef(kHardwareCase), {fabricTarget()}}},
      {EvaluationConditionKind::ActivityBinding,
       {caseRef(kHardwareCase), {fabricTarget(), fabricTarget()}}},
  };
  return patterns;
}

const CaseSubjectRoleDescriptor kSystemRoles[] = {
    {kRole0, "deployment", SubjectRoleCardinality::ExactlyOne,
     kDeploymentSchemas, nullptr},
    {kRole1, "gem5_simulation_binding", SubjectRoleCardinality::ExactlyOne,
     kGem5Schemas, &verifyGem5Binding},
};
const EvaluationCaseSignatureDescriptor kSystemSignature{
    builtinEvaluationCaseKind(kSystemCase),
    "system_simulation",
    "One exact Deployment executed through one exact gem5 simulation binding.",
    kSystemRoles,
    ArtifactRequirement::Required,
    kWorkloadSchemas,
    ArtifactRequirement::Required,
    kRuntimeInputSchemas,
    &verifySystemWorkload,
    AbsentReferenceCycle{},
    {}};

const CaseSubjectRoleDescriptor kHardwareRoles[] = {
    {kRole0, "fabric", SubjectRoleCardinality::ExactlyOne, kFabricSchemas,
     nullptr},
};
const EvaluationCaseSignatureDescriptor kHardwareSignature{
    builtinEvaluationCaseKind(kHardwareCase),
    "fabric_hardware_analysis",
    "One exact Fabric analyzed without a software workload.",
    kHardwareRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    hardwareConditions()};

const CaseSubjectRoleDescriptor kFpaCalibrationRoles[] = {
    {kRole0, "fpa_parameter_bundle", SubjectRoleCardinality::ExactlyOne,
     kParameterSchemas, &verifyFpaParameterSubject},
    {kRole1, "ground_truth_evidence", SubjectRoleCardinality::OneOrMore,
     kEvidenceSchemas, &verifyFpaGroundTruthSubject},
};
const EvaluationCaseSignatureDescriptor kFpaCalibrationSignature{
    builtinEvaluationCaseKind(kFpaCalibrationCase),
    "fpa_model_parameter_calibration",
    "One FPA parameter bundle calibrated against exact evidence.",
    kFpaCalibrationRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

const CaseSubjectRoleDescriptor kRuntimeCalibrationRoles[] = {
    {kRole0, "system_runtime_parameter_bundle",
     SubjectRoleCardinality::ExactlyOne, kParameterSchemas,
     &verifySystemRuntimeParameterSubject},
    {kRole1, "ground_truth_evidence", SubjectRoleCardinality::OneOrMore,
     kEvidenceSchemas, &verifySystemRuntimeGroundTruthSubject},
};
const EvaluationCaseSignatureDescriptor kRuntimeCalibrationSignature{
    builtinEvaluationCaseKind(kRuntimeCalibrationCase),
    "system_runtime_model_parameter_calibration",
    "One System Runtime parameter bundle calibrated against exact evidence.",
    kRuntimeCalibrationRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

const CaseSubjectRoleDescriptor kMappedRtlRoles[] = {
    {kRole0, "hardware_implementation", SubjectRoleCardinality::ExactlyOne,
     kHardwareSchemas, nullptr},
    {kRole1, "deployment", SubjectRoleCardinality::ExactlyOne,
     kDeploymentSchemas, nullptr},
};
const EvaluationCaseSignatureDescriptor kMappedRtlSignature{
    builtinEvaluationCaseKind(kMappedRtlCase),
    "mapped_rtl_simulation",
    "One mapped Spatial workload executed by its exact RTL implementation.",
    kMappedRtlRoles,
    ArtifactRequirement::Required,
    kWorkloadSchemas,
    ArtifactRequirement::Required,
    kRuntimeInputSchemas,
    &verifyMappedRtlWorkload,
    ExactSubjectCycle{mappingRootType(), &resolveMappedRtlCycle},
    {}};

template <BuiltinEvaluationModel Model> struct EmptyModelConfig {};

llvm::StringRef modelConfigOwner(BuiltinEvaluationModel model) {
  switch (model) {
  case BuiltinEvaluationModel::FpaModelParameterCalibration:
    return "fpa_model_parameter_calibration";
  case BuiltinEvaluationModel::StructuredFabricCalibratedFpa:
    return "structured_fabric_calibrated_fpa";
  case BuiltinEvaluationModel::CanonicalDataflowFabricCalibratedFpa:
    return "canonical_dataflow_fabric_calibrated_fpa";
  case BuiltinEvaluationModel::FabricCalibratedFpa:
    return "fabric_calibrated_fpa";
  case BuiltinEvaluationModel::SystemRuntimeModelParameterCalibration:
    return "system_runtime_model_parameter_calibration";
  case BuiltinEvaluationModel::Gem5CgraSystemRuntimePredictor:
    return "gem5_cgra_system_runtime_predictor";
  case BuiltinEvaluationModel::Gem5SystemDfg:
    return "gem5_system_dfg";
  case BuiltinEvaluationModel::Gem5SystemCgra:
    return "gem5_system_cgra";
  case BuiltinEvaluationModel::Gem5SystemRtl:
    return "gem5_system_rtl";
  case BuiltinEvaluationModel::OpenRoadRoutedStaticFpa:
    return "openroad_routed_static_fpa";
  case BuiltinEvaluationModel::MappedRtlSimulator:
    return "mapped_rtl_simulator";
  default:
    llvm_unreachable("built-in model has no production config contract");
  }
}

template <BuiltinEvaluationModel Model>
llvm::ArrayRef<std::uint8_t> emptyConfigSchema() {
  static const std::string schema =
      ("loom.evaluation." + modelConfigOwner(Model) + ".config.1.0").str();
  return {reinterpret_cast<const std::uint8_t *>(schema.data()), schema.size()};
}

template <BuiltinEvaluationModel Model>
llvm::Expected<OwnerValue> projectEmptyConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyModelConfig<Model>{});
}

template <BuiltinEvaluationModel Model>
llvm::Expected<std::vector<std::uint8_t>>
encodeEmptyConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyModelConfig<Model>>())
    return invalid("model config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

template <BuiltinEvaluationModel Model>
llvm::Expected<OwnerValue> adoptEmptyConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                            const ComponentViewDigest &) {
  if (!bytes.empty())
    return invalid("fixed model config must be empty");
  return OwnerValue::get(EmptyModelConfig<Model>{});
}

template <BuiltinEvaluationModel Model>
const ResolvedModelConfigViewContract &emptyConfig() {
  static const ResolvedModelConfigViewContract contract{
      emptyConfigSchema<Model>(), &projectEmptyConfig<Model>,
      &encodeEmptyConfig<Model>, &adoptEmptyConfig<Model>};
  return contract;
}

template <BuiltinEvaluationModel Model> struct ProviderBuildConfig {
  std::string stableBuildIdentity;
};

template <BuiltinEvaluationModel Model>
llvm::Expected<OwnerValue> projectProviderBuild(const ResolvedConfig &) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      "provider build binding is unavailable in ResolvedConfig");
}

template <BuiltinEvaluationModel Model>
llvm::Expected<std::vector<std::uint8_t>>
encodeProviderBuild(const OwnerValue &value) {
  const auto *config = value.getIf<ProviderBuildConfig<Model>>();
  if (!config || config->stableBuildIdentity.empty())
    return invalid("provider build config is missing its stable identity");
  return std::vector<std::uint8_t>(config->stableBuildIdentity.begin(),
                                   config->stableBuildIdentity.end());
}

template <BuiltinEvaluationModel Model>
llvm::Expected<OwnerValue>
adoptProviderBuild(llvm::ArrayRef<std::uint8_t> bytes,
                   const ComponentViewDigest &) {
  if (bytes.empty())
    return invalid("provider build config is empty");
  std::string identity(bytes.begin(), bytes.end());
  if (!llvm::all_of(identity, [](unsigned char character) {
        return character >= 0x21 && character <= 0x7e;
      }))
    return invalid("provider build identity is not canonical ASCII");
  return OwnerValue::get(ProviderBuildConfig<Model>{std::move(identity)});
}

template <BuiltinEvaluationModel Model>
const ResolvedModelConfigViewContract &providerBuildConfig() {
  static const ResolvedModelConfigViewContract contract{
      emptyConfigSchema<Model>(), &projectProviderBuild<Model>,
      &encodeProviderBuild<Model>, &adoptProviderBuild<Model>};
  return contract;
}

const ScopeFormRef kWholeCaseScopes[] = {kWholeCase};
constexpr std::uint8_t kPoint = observationFormMask(ObservationForm::Point);
const MetricCapability kFpaMetrics[] = {
    {MetricKind::LimitingClockFrequency, kWholeCaseScopes, kPoint},
    {MetricKind::TotalArea, kWholeCaseScopes, kPoint},
    {MetricKind::DynamicPower, kWholeCaseScopes, kPoint},
    {MetricKind::LeakagePower, kWholeCaseScopes, kPoint},
};
const MetricCapability kRuntimeMetric[] = {
    {MetricKind::Runtime, kWholeCaseScopes, kPoint}};
const MetricCapability kRuntimeErrorMetric[] = {
    {MetricKind::RuntimePredictionError, kWholeCaseScopes, kPoint}};
const MetricCapability kFpaErrorMetrics[] = {
    {MetricKind::LimitingClockFrequencyPredictionError, kWholeCaseScopes,
     kPoint},
    {MetricKind::TotalAreaPredictionError, kWholeCaseScopes, kPoint},
    {MetricKind::DynamicPowerPredictionError, kWholeCaseScopes, kPoint},
    {MetricKind::LeakagePowerPredictionError, kWholeCaseScopes, kPoint},
};
const MetricCapability kCycleMetric[] = {
    {MetricKind::CycleCount, kWholeCaseScopes, kPoint}};

const std::vector<ModelConditionCapability> &hardwareConditionCapabilities() {
  static const std::vector<ModelConditionCapability> capabilities = [] {
    std::vector<ModelConditionCapability> result;
    for (const ConditionApplicabilityPattern &pattern : hardwareConditions())
      result.push_back({pattern, ConditionDisposition::Consumed});
    return result;
  }();
  return capabilities;
}

const std::vector<ModelConditionCapability> &openRoadConditionCapabilities() {
  static const std::vector<ModelConditionCapability> capabilities = [] {
    std::vector<ModelConditionCapability> result;
    const auto patterns =
        models::hardwareImplementationPhysicalBaseConditionPatterns();
    for (std::size_t index = 0; index < patterns.size(); ++index) {
      const bool required = index < 4;
      result.push_back({patterns[index], required
                                             ? ConditionDisposition::Required
                                             : ConditionDisposition::Consumed});
    }
    return result;
  }();
  return capabilities;
}

llvm::ArrayRef<ModelConditionCapability> runtimeErrorConditions() {
  static const std::array<ModelConditionCapability, 1> capabilities = {{
      {metricDescriptor(MetricKind::RuntimePredictionError)
           .requiredRequestConditionPatterns.front(),
       ConditionDisposition::Required},
  }};
  return capabilities;
}

llvm::ArrayRef<ModelConditionCapability> fpaErrorConditions() {
  static const std::array<ModelConditionCapability, 1> capabilities = {{
      {metricDescriptor(MetricKind::LimitingClockFrequencyPredictionError)
           .requiredRequestConditionPatterns.front(),
       ConditionDisposition::Required},
  }};
  return capabilities;
}

const std::vector<ModelConditionCapability> &
structuredFpaPredictionConditions() {
  static const std::vector<ModelConditionCapability> capabilities = [] {
    std::vector<ModelConditionCapability> result;
    const EvaluationCaseSignatureRef reference =
        caseRef(BuiltinEvaluationCase::StructuredProgramWithFabric);
    for (const ConditionApplicabilityPattern &pattern :
         reference.descriptor()->permittedBaseConditions)
      result.push_back({pattern, ConditionDisposition::Consumed});
    return result;
  }();
  return capabilities;
}

const std::vector<ModelConditionCapability> &dataflowFpaPredictionConditions() {
  static const std::vector<ModelConditionCapability> capabilities = [] {
    std::vector<ModelConditionCapability> result;
    const EvaluationCaseSignatureRef reference =
        caseRef(BuiltinEvaluationCase::CanonicalDataflowWithFabric);
    for (const ConditionApplicabilityPattern &pattern :
         reference.descriptor()->permittedBaseConditions)
      result.push_back({pattern, ConditionDisposition::Consumed});
    return result;
  }();
  return capabilities;
}

const ModelInputSlotDescriptor kFpaParameterInputs[] = {
    {kParameterInput, "model_parameter_bundle", kParameterSchemas,
     ArtifactCollectionCardinality::ExactlyOne, nullptr,
     models::fpaModelParameterContractRef()}};
const ModelInputSlotDescriptor kSystemRuntimeParameterInputs[] = {
    {kParameterInput, "model_parameter_bundle", kParameterSchemas,
     ArtifactCollectionCardinality::ExactlyOne, nullptr,
     models::systemRuntimeModelParameterContractRef()}};

const ModelOutputSlotDescriptor kExecutionOutputs[] = {{
    kExecutionOutput,
    "simulation_execution",
    &sim::simulationExecutionSchema,
    {ArtifactCollectionCardinality::ExactlyOne,
     ArtifactCollectionCardinality::Forbidden,
     ArtifactCollectionCardinality::Forbidden,
     ArtifactCollectionCardinality::Forbidden},
}};

const ModeledPhenomenon kHardwarePhenomena[] = {
    ModeledPhenomenon::SpatialResources,
    ModeledPhenomenon::PhysicalImplementation};
const ModeledPhenomenon kSystemPhenomena[] = {
    ModeledPhenomenon::StructuredProgram, ModeledPhenomenon::CanonicalDataflow,
    ModeledPhenomenon::SystemMemoryHierarchy, ModeledPhenomenon::Coherence};
const ModeledPhenomenon kRtlPhenomena[] = {ModeledPhenomenon::CanonicalDataflow,
                                           ModeledPhenomenon::RTLBehavior};

llvm::ArrayRef<EvaluationModelDescriptor> builtinModelDescriptors() {
  static const EvaluationModelDescriptor descriptors[] = {
      {builtinEvaluationModelKind(
           BuiltinEvaluationModel::FpaModelParameterCalibration),
       "fpa_model_parameter_calibration",
       "loom.fpa.model_parameter_calibration.v1",
       caseRef(kFpaCalibrationCase),
       fpaErrorConditions(),
       kFpaErrorMetrics,
       {},
       {},
       {},
       emptyConfig<BuiltinEvaluationModel::FpaModelParameterCalibration>(),
       {},
       EvaluationExecutionMethod::Analytic,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::InProcess},
      {builtinEvaluationModelKind(
           BuiltinEvaluationModel::StructuredFabricCalibratedFpa),
       "structured_fabric_calibrated_fpa",
       "loom.structured_fabric.calibrated_fpa.v1",
       caseRef(BuiltinEvaluationCase::StructuredProgramWithFabric),
       structuredFpaPredictionConditions(),
       kFpaMetrics,
       {},
       kFpaParameterInputs,
       {},
       emptyConfig<BuiltinEvaluationModel::StructuredFabricCalibratedFpa>(),
       kHardwarePhenomena,
       EvaluationExecutionMethod::Analytic,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::InProcess},
      {builtinEvaluationModelKind(
           BuiltinEvaluationModel::CanonicalDataflowFabricCalibratedFpa),
       "canonical_dataflow_fabric_calibrated_fpa",
       "loom.canonical_dataflow_fabric.calibrated_fpa.v1",
       caseRef(BuiltinEvaluationCase::CanonicalDataflowWithFabric),
       dataflowFpaPredictionConditions(),
       kFpaMetrics,
       {},
       kFpaParameterInputs,
       {},
       emptyConfig<
           BuiltinEvaluationModel::CanonicalDataflowFabricCalibratedFpa>(),
       kHardwarePhenomena,
       EvaluationExecutionMethod::Analytic,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::InProcess},
      {builtinEvaluationModelKind(BuiltinEvaluationModel::FabricLowConfidence),
       "fabric_low_confidence",
       "loom.fabric.low_confidence.v1",
       caseRef(kHardwareCase),
       hardwareConditionCapabilities(),
       kFpaMetrics,
       {},
       {},
       {},
       models::detail::emptyLowConfidenceConfigView(),
       kHardwarePhenomena,
       EvaluationExecutionMethod::Analytic,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::InProcess},
      {builtinEvaluationModelKind(BuiltinEvaluationModel::FabricCalibratedFpa),
       "fabric_calibrated_fpa",
       "loom.fabric.calibrated_fpa.v1",
       caseRef(kHardwareCase),
       hardwareConditionCapabilities(),
       kFpaMetrics,
       {},
       kFpaParameterInputs,
       {},
       emptyConfig<BuiltinEvaluationModel::FabricCalibratedFpa>(),
       kHardwarePhenomena,
       EvaluationExecutionMethod::Analytic,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::InProcess},
      {builtinEvaluationModelKind(
           BuiltinEvaluationModel::SystemRuntimeModelParameterCalibration),
       "system_runtime_model_parameter_calibration",
       "loom.system_runtime.calibration.v1",
       caseRef(kRuntimeCalibrationCase),
       runtimeErrorConditions(),
       kRuntimeErrorMetric,
       {},
       {},
       {},
       emptyConfig<
           BuiltinEvaluationModel::SystemRuntimeModelParameterCalibration>(),
       {},
       EvaluationExecutionMethod::Analytic,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::InProcess},
      {builtinEvaluationModelKind(
           BuiltinEvaluationModel::Gem5CgraSystemRuntimePredictor),
       "gem5_cgra_system_runtime_predictor",
       "loom.gem5_cgra.system_runtime_predictor.v1",
       caseRef(kSystemCase),
       {},
       kRuntimeMetric,
       {},
       kSystemRuntimeParameterInputs,
       {},
       emptyConfig<BuiltinEvaluationModel::Gem5CgraSystemRuntimePredictor>(),
       kSystemPhenomena,
       EvaluationExecutionMethod::Analytic,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::InProcess},
      {builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemDfg),
       "gem5_system_dfg",
       "loom.gem5.system_dfg.v1",
       caseRef(kSystemCase),
       {},
       {},
       {},
       {},
       kExecutionOutputs,
       emptyConfig<BuiltinEvaluationModel::Gem5SystemDfg>(),
       kSystemPhenomena,
       EvaluationExecutionMethod::Simulation,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::ExternalPrepareImport},
      {builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemCgra),
       "gem5_system_cgra",
       "loom.gem5.system_cgra.v2",
       caseRef(kSystemCase),
       {},
       kRuntimeMetric,
       {},
       {},
       kExecutionOutputs,
       emptyConfig<BuiltinEvaluationModel::Gem5SystemCgra>(),
       kSystemPhenomena,
       EvaluationExecutionMethod::Simulation,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::ExternalPrepareImport},
      {builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemRtl),
       "gem5_system_rtl",
       "loom.gem5.system_rtl.v2",
       caseRef(kSystemCase),
       {},
       kRuntimeMetric,
       {},
       {},
       kExecutionOutputs,
       models::mappedRtlSimulationConfigViewContract(),
       kSystemPhenomena,
       EvaluationExecutionMethod::Simulation,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::ExternalPrepareImport},
      {builtinEvaluationModelKind(
           BuiltinEvaluationModel::OpenRoadRoutedStaticFpa),
       "openroad_routed_static_fpa",
       "loom.eda.openroad.routed_static_fpa.v1",
       models::hardwareImplementationPhysicalCaseSignatureRef(),
       openRoadConditionCapabilities(),
       kFpaMetrics,
      {},
      {},
      {},
      models::openRoadStaticFpaConfigViewContract(),
       kHardwarePhenomena,
       EvaluationExecutionMethod::ToolMeasurement,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::ExternalPrepareImport},
      {builtinEvaluationModelKind(BuiltinEvaluationModel::MappedRtlSimulator),
       "mapped_rtl_simulator",
       models::mappedRtlSimulatorSemanticIdentity,
       caseRef(kMappedRtlCase),
       {},
       kCycleMetric,
       {},
       {},
       kExecutionOutputs,
       models::mappedRtlSimulationConfigViewContract(),
       kRtlPhenomena,
       EvaluationExecutionMethod::Simulation,
       {},
       DeterminismContract::Deterministic,
       {},
       ProviderForm::ExternalPrepareImport},
  };
  return descriptors;
}

} // namespace

llvm::Error registerProductionEvaluationRegistry() {
  if (llvm::Error error = runtime::registerBuiltinGem5ModelContracts())
    return error;
  if (llvm::Error error = models::registerStructuredFabricAnalyticModel())
    return error;
  if (llvm::Error error =
          models::registerCanonicalDataflowFabricAnalyticModel())
    return error;
  if (llvm::Error error = models::registerStructuredProgramFunctionalModel())
    return error;
  if (llvm::Error error = models::registerDfgSimulationModel())
    return error;
  if (llvm::Error error = models::registerCgraSimulationModel())
    return error;
  if (llvm::Error error = models::registerSimulationComparisonModel())
    return error;
  if (llvm::Error error = models::registerCanonicalDataflowFunctionalModel())
    return error;
  if (llvm::Error error = models::registerCadenceVoltusStaticRailModel())
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kSystemSignature))
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kHardwareSignature))
    return error;
  if (llvm::Error error =
          registerEvaluationCaseSignature(kFpaCalibrationSignature))
    return error;
  if (llvm::Error error =
          registerEvaluationCaseSignature(kRuntimeCalibrationSignature))
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kMappedRtlSignature))
    return error;
  for (const EvaluationModelDescriptor &model : builtinModelDescriptors())
    if (llvm::Error error = registerEvaluationModelDescriptor(model))
      return error;
  if (llvm::Error error = models::registerFpaModelParameterContract())
    return error;
  if (llvm::Error error = models::registerSystemRuntimeModelParameterContract())
    return error;
  if (llvm::Error error = models::registerFabricLowConfidenceProvider())
    return error;
  if (llvm::Error error = models::registerCalibratedFpaProviders())
    return error;
  if (llvm::Error error = models::registerSystemRuntimePredictorProvider())
    return error;
  if (llvm::Error error = models::registerPredictionCalibrationProviders())
    return error;
  if (llvm::Error error = models::registerGem5SystemDfgProvider())
    return error;
  if (llvm::Error error = models::registerGem5SystemCgraProvider())
    return error;
  return models::registerGem5SystemRtlProvider();
}

EvaluationCaseSignatureRef systemSimulationCaseSignatureRef() {
  return caseRef(kSystemCase);
}

EvaluationCaseSignatureRef fabricHardwareAnalysisCaseSignatureRef() {
  return caseRef(kHardwareCase);
}

EvaluationCaseSignatureRef systemRuntimeCalibrationCaseSignatureRef() {
  return caseRef(kRuntimeCalibrationCase);
}

EvaluationCaseSignatureRef mappedRtlSimulationCaseSignatureRef() {
  return caseRef(kMappedRtlCase);
}

EvaluationCaseSignatureRef
builtinEvaluationCaseSignatureRef(BuiltinEvaluationCase evaluationCase) {
  return caseRef(evaluationCase);
}

llvm::Expected<EvaluationModelDescriptorRef>
builtinEvaluationModelDescriptorRef(BuiltinEvaluationModel model) {
  return EvaluationModelDescriptorRef::get(evaluationSchemaVersion(),
                                           builtinEvaluationModelKind(model));
}

} // namespace loom::evaluation
