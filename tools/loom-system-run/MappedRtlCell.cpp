#include "MappedRtlCell.h"

#include "Common/ArtifactText.h"
#include "Common/ExecutionControl.h"
#include "Config/ResolvedConfig.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "Deployment/Package.h"
#include "EDA/Adapters/OpenSource/MappedRtlRuntimePlatform.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/MappedRtlSimulation.h"
#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/LocalConfig.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/ShellProbe.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/MaterializationDiagnostics.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"

#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::system_run;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_cell_invalid: " + message);
}

llvm::Error ioError(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                 "mapped_rtl_cell_io_error: " + message);
}

std::string child(llvm::StringRef parent, llvm::StringRef name) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, name);
  return path.str().str();
}

llvm::Expected<evaluation::CaseArtifactResolution> buildMappedRtlResolution(
    const SpatialInvocationCase &invocation,
    const loom::deployment::FinalizedDeployment &deployment,
    const loom::ArtifactRootReference &hardwareImplementation,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  auto package = loom::deployment::deriveDeploymentPackageClosure(
      deployment, artifacts, blobs);
  if (!package)
    return package.takeError();
  if (!llvm::is_contained(package->artifacts(), invocation.dataflow) ||
      !llvm::is_contained(package->artifacts(), hardwareImplementation))
    return invalid("mapped RTL Deployment omits an invocation owner");
  std::vector<loom::evaluation::CaseArtifactResolution::Entry> entries;
  entries.reserve(package->artifacts().size() + 2);
  for (const auto &root : package->artifacts())
    entries.push_back({root, {}});
  for (auto &entry : entries)
    if (entry.artifact == deployment.reference()) {
      entry.dependencyClosure.clear();
      for (const auto &root : package->artifacts())
        if (root != deployment.reference())
          entry.dependencyClosure.push_back(root);
      break;
    }
  entries.push_back({invocation.workload, {invocation.dataflow}});
  entries.push_back(
      {invocation.runtimeInput, {invocation.dataflow, invocation.workload}});
  return loom::evaluation::CaseArtifactResolution::get(std::move(entries));
}

} // namespace

llvm::Expected<deployment::FinalizedDeployment>
loom::system_run::deriveMappedRtlDeployment(
    const deployment::FinalizedDeployment &source,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  loom::hardware::rtl::RtlMaterializationStageTracker derivationStage(
      "mapped_rtl_deployment_derivation",
      loom::formatArtifactIdentityHex(source.reference().artifact));
  std::vector<loom::deployment::DeploymentHardwareBinding> hardwareBindings;
  hardwareBindings.reserve(source.deployment().hardwareBindings().size());
  for (const auto indexedBinding :
       llvm::enumerate(source.deployment().hardwareBindings())) {
    const auto &binding = indexedBinding.value();
    const std::string bindingKey =
        loom::formatArtifactIdentityHex(source.reference().artifact) +
        ":binding:" + std::to_string(indexedBinding.index());
    auto model = loom::hardware::importHardwareImplementation(
        binding.hardwareImplementation, artifacts, blobs);
    if (!model)
      return model.takeError();
    auto abi = loom::hardware::importConfigurationABI(
        model->implementation().configurationAbi(), artifacts);
    if (!abi)
      return abi.takeError();
    loom::hardware::rtl::RtlMaterializationStageTracker implementationStage(
        "portable_hardware_implementation_finalization", bindingKey);
    auto rtl = loom::hardware::rtl::
        finalizePortableSpatialCoreHardwareImplementation(
            *abi, model->implementation().subject(), std::nullopt, artifacts,
            blobs);
    if (!rtl)
      return rtl.takeError();
    implementationStage.finish();
    loom::hardware::rtl::RtlMaterializationStageTracker runtimeStage(
        "mapped_rtl_runtime_binding_finalization", bindingKey);
    auto runtimeBinding =
        loom::eda::open_source::finalizeMappedRtlRuntimePlatformBinding(
            *rtl, artifacts, blobs);
    if (!runtimeBinding)
      return runtimeBinding.takeError();
    runtimeStage.finish();
    hardwareBindings.push_back({rtl->reference(), runtimeBinding->reference()});
  }
  loom::hardware::rtl::RtlMaterializationStageTracker deploymentStage(
      "mapped_rtl_deployment_build",
      loom::formatArtifactIdentityHex(source.reference().artifact));
  auto deployment = loom::deployment::buildDeployment(
      {source.deployment().systemMapping(), source.deployment().hostProgram(),
       source.deployment().instructionCoreBinaries().vec(),
       std::move(hardwareBindings),
       source.deployment().staticMemoryImages().vec()},
      artifacts, blobs);
  if (!deployment)
    return deployment.takeError();
  deploymentStage.finish();
  derivationStage.finish();
  return deployment;
}

llvm::Expected<MappedRtlCellEvidence> loom::system_run::executeMappedRtlCell(
    const SpatialInvocationCase &invocation,
    const deployment::FinalizedDeployment &deployment,
    llvm::StringRef bundleRoot, const MappedRtlProviderOptions &options,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto inputs = loom::sim::importSpatialSimulationInputs(
      invocation.workload, invocation.runtimeInput, artifacts);
  if (!inputs)
    return inputs.takeError();
  const auto *workload = inputs->workload.spatial();
  if (!workload)
    return invalid("mapped RTL invocation workload is not Spatial");
  auto selection = loom::deployment::resolveDeploymentSpatialLaunchSelection(
      deployment, workload->launchRef, invocation.denseCoordinates, artifacts,
      blobs);
  if (!selection)
    return selection.takeError();
  if (selection->spatialMapping != invocation.spatialMapping)
    return invalid("mapped RTL Deployment selected another SpatialMapping");
  auto resolution = buildMappedRtlResolution(invocation, deployment,
                                             selection->hardwareImplementation,
                                             artifacts, blobs);
  if (!resolution)
    return resolution.takeError();

  const auto &provider = loom::external_tool::verilatorProvider();
  const std::string probePath = child(
      bundleRoot, ("spatial-rtl-probe-" + llvm::Twine(invocation.ordinal)).str());
  std::error_code directoryError;
  if (!std::filesystem::create_directory(probePath, directoryError) ||
      directoryError)
    return ioError("cannot create mapped RTL tool probe directory");
  loom::external_tool::LocalToolConfig local;
  local.runtimePolicy = loom::external_tool::RuntimePolicy::Host;
  loom::external_tool::ShellToolBindingProbe probe(probePath,
                                                   provider.versionProbe);
  auto resolvedTool = loom::external_tool::resolveToolBinding(
      provider.binding, local,
      loom::external_tool::captureToolEnvironment(provider.binding), probe);
  if (!resolvedTool)
    return resolvedTool.takeError();
  local.tools[provider.binding.key].binding.executable =
      resolvedTool->executable;
  local.tools[provider.binding.key].providerOptions["build_jobs"] =
      options.buildJobs;
  local.tools[provider.binding.key].providerOptions["build_workers"] =
      options.buildWorkers;
  local.tools[provider.binding.key].providerOptions["model_threads"] =
      options.modelThreads;

  auto subjects = loom::evaluation::EvaluationSubjectBindings::get(
      {{loom::evaluation::models::mappedRtlHardwareImplementationSubjectRole(),
        {selection->hardwareImplementation}},
       {loom::evaluation::models::mappedRtlDeploymentSubjectRole(),
        {deployment.reference()}}});
  if (!subjects)
    return subjects.takeError();
  auto evaluationCase = loom::evaluation::EvaluationCase::get(
      loom::evaluation::mappedRtlSimulationCaseSignatureRef(),
      std::move(*subjects), invocation.workload, invocation.runtimeInput, {},
      *resolution, artifacts, blobs);
  if (!evaluationCase)
    return evaluationCase.takeError();
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.evaluation.mappedRtlSimulator =
      loom::evaluation::models::MappedRtlSimulatorBinding{
          resolvedTool->version};
  auto model = loom::evaluation::ResolvedModelBinding::project(
      loom::evaluation::models::mappedRtlSimulatorModelDescriptorRef(), {},
      config);
  if (!model)
    return model.takeError();
  auto request = loom::evaluation::EvaluationRequest::get(
      *evaluationCase, {}, {}, std::move(*model), 0, *resolution, artifacts,
      blobs);
  if (!request)
    return request.takeError();
  auto requestReference =
      loom::evaluation::publishEvaluationRequest(*request, artifacts);
  if (!requestReference)
    return requestReference.takeError();
  if (*requestReference !=
      loom::evaluation::evaluationRequestReference(*request))
    return invalid("mapped RTL Request publication changed its identity");
  const std::string bundlePath =
      child(bundleRoot, ("spatial-rtl-" + llvm::Twine(invocation.ordinal)).str());
  const std::string materializationKey = loom::formatArtifactIdentityHex(
      loom::evaluation::evaluationRequestReference(*request).artifact);
  loom::hardware::rtl::RtlMaterializationStageTracker preparationStage(
      "mapped_rtl_provider_preparation", materializationKey);
  auto preparation = loom::evaluation::prepareEvaluationModelInvocation(
      *request, *resolution, artifacts, blobs, {std::move(local), bundlePath});
  if (!preparation)
    return preparation.takeError();
  preparationStage.finish();
  const auto *prepared =
      std::get_if<loom::evaluation::EvaluationModelPreparedInvocation>(
          &*preparation);
  if (!prepared)
    return invalid("mapped RTL is unsupported for the materialized invocation");
  const auto &external = prepared->externalInvocation();
  {
    loom::hardware::rtl::RtlMaterializationStageTracker releaseStage(
        "mapped_rtl_provider_release", materializationKey);
    (void)loom::releaseUnusedProcessMemory();
    releaseStage.finish();
  }
  loom::hardware::rtl::RtlMaterializationStageTracker executionStage(
      "verilator_execution", materializationKey);
  auto execution =
      loom::external_tool::executeExternalToolInvocationBundleObserved(
          external);
  if (!execution)
    return execution.takeError();
  executionStage.finish();
  if (execution->exitCode != 0)
    return invalid("spatial-rtl external invocation exited with status " +
                   llvm::Twine(execution->exitCode));
  auto evidence = loom::evaluation::importEvaluationModelInvocation(
      *request, *resolution, *prepared, *execution, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  return MappedRtlCellEvidence{std::move(*request), std::move(*resolution),
                               std::move(*evidence)};
}

