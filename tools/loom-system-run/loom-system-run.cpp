#include "MappedRtlCell.h"
#include "SpatialInvocationCase.h"
#include "SystemRunError.h"

#include "Application/DeploymentRuntime.h"
#include "Application/Package.h"
#include "Application/ProductOracleEvaluation.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ProductionOwners.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Deployment/Deployment.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Deployment/Package.h"
#include "EDA/Adapters/OpenSource/MappedRtlExecution.h"
#include "EDA/Adapters/OpenSource/MappedRtlSimulation.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/LocalConfig.h"
#include "ExternalTool/Provider.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Configuration/ConfigurationDiagnostics.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Runtime/FabricModelPlatform.h"
#include "Runtime/FabricModelRuntimeProvider.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5RootEventControl.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Runtime/Gem5SystemExecution.h"
#include "Runtime/SpatialInvocationWire.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialInvocation.h"
#include "Simulator/SpatialObservationComparison.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

constexpr std::size_t invocationSystemMappingEntries = 1;
constexpr std::size_t invocationConfigurationProjectionEntries = 1;
constexpr std::size_t invocationArtifactImportEntries = 16;

llvm::cl::opt<std::string>
    deploymentPackage("deployment-package",
                      llvm::cl::desc("Input Application package"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    outputWorkspace("output", llvm::cl::desc("New execution workspace"),
                    llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    gem5Readiness("gem5-readiness",
                  llvm::cl::desc("Pinned gem5 readiness JSON"),
                  llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::int64_t>
    expectedI32("expected-i32",
                llvm::cl::desc("Independent expected i32 result"));
llvm::cl::opt<bool>
    mappedRtl("mapped-rtl",
              llvm::cl::desc(
                  "Execute each materialized Spatial invocation as mapped RTL"),
              llvm::cl::init(false));
llvm::cl::opt<loom::eda::open_source::MappedRtlHdlSimulator> mappedRtlSimulator(
    "mapped-rtl-simulator",
    llvm::cl::desc("HDL simulator that compiles and runs the mapped RTL"),
    llvm::cl::values(
        clEnumValN(loom::eda::open_source::MappedRtlHdlSimulator::Verilator,
                   loom::eda::open_source::mappedRtlHdlSimulatorSpelling(
                       loom::eda::open_source::MappedRtlHdlSimulator::Verilator)
                       .data(),
                   "Verilator, flat multithreaded model"),
        clEnumValN(loom::eda::open_source::MappedRtlHdlSimulator::Vcs,
                   loom::eda::open_source::mappedRtlHdlSimulatorSpelling(
                       loom::eda::open_source::MappedRtlHdlSimulator::Vcs)
                       .data(),
                   "Synopsys VCS, event-driven four-state model"),
        clEnumValN(loom::eda::open_source::MappedRtlHdlSimulator::Xcelium,
                   loom::eda::open_source::mappedRtlHdlSimulatorSpelling(
                       loom::eda::open_source::MappedRtlHdlSimulator::Xcelium)
                       .data(),
                   "Cadence Xcelium, event-driven four-state model")),
    llvm::cl::init(loom::eda::open_source::MappedRtlHdlSimulator::Verilator));
llvm::cl::opt<std::uint64_t> mappedRtlBuildJobs(
    "mapped-rtl-build-jobs",
    llvm::cl::desc("Parallel compilation jobs of the mapped RTL simulator"),
    llvm::cl::init(loom::eda::open_source::mappedRtlDefaultBuildJobs));
llvm::cl::opt<std::uint64_t> mappedRtlBuildWorkers(
    "mapped-rtl-build-workers",
    llvm::cl::desc("Mapped RTL concurrent bundle build limit"),
    llvm::cl::init(loom::eda::open_source::mappedRtlDefaultBuildWorkers));
llvm::cl::opt<std::uint64_t> mappedRtlModelThreads(
    "mapped-rtl-model-threads",
    llvm::cl::desc("Simulation threads of the Verilated mapped RTL model"),
    llvm::cl::init(loom::eda::open_source::mappedRtlDefaultModelThreads));
llvm::cl::opt<std::string> mappedRtlLocalToolConfigPath(
    "mapped-rtl-local-tool-config",
    llvm::cl::desc("Explicit local tool configuration for mapped RTL"),
    llvm::cl::value_desc("path"));
llvm::cl::opt<std::string> mappedRtlProviderBuild(
    "mapped-rtl-provider-build",
    llvm::cl::desc("Exact mapped-RTL simulator version-probe line"),
    llvm::cl::value_desc("build"));
llvm::cl::opt<std::string> spatialCgraProfileOutput(
    "spatial-cgra-profile-output",
    llvm::cl::desc("Write invocation-local Spatial CGRA qualification data"),
    llvm::cl::value_desc("path"));
llvm::cl::opt<std::uint64_t> spatialCgraWarmupRuns(
    "spatial-cgra-warmup-runs",
    llvm::cl::desc("Prepared Spatial CGRA warmup attempts"), llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> spatialCgraMeasurementRuns(
    "spatial-cgra-measurement-runs",
    llvm::cl::desc("Prepared Spatial CGRA measured attempts"),
    llvm::cl::init(0));

using loom::system_run::invalid;

llvm::Error ioError(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                 "system_run_io_error: " + message);
}

llvm::Expected<loom::system_run::MappedRtlProviderOptions>
loadMappedRtlProviderOptions() {
  if (mappedRtlLocalToolConfigPath.empty())
    return invalid("--mapped-rtl-local-tool-config is required with "
                   "--mapped-rtl");
  if (mappedRtlProviderBuild.empty())
    return invalid("--mapped-rtl-provider-build is required with "
                   "--mapped-rtl");
  using Simulator = loom::eda::open_source::MappedRtlHdlSimulator;
  if (mappedRtlSimulator != Simulator::Verilator &&
      (mappedRtlBuildWorkers.getNumOccurrences() ||
       mappedRtlModelThreads.getNumOccurrences()))
    return invalid("mapped-RTL bundle build workers and model threads require "
                   "Verilator");
  if (mappedRtlSimulator == Simulator::Xcelium &&
      mappedRtlBuildJobs.getNumOccurrences())
    return invalid("mapped-RTL compilation jobs are not supported by Xcelium");
  auto local =
      loom::external_tool::loadLocalToolConfig(mappedRtlLocalToolConfigPath);
  if (!local)
    return local.takeError();
  loom::system_run::MappedRtlProviderOptions options{
      mappedRtlSimulator,
      mappedRtlBuildJobs,
      mappedRtlBuildWorkers,
      mappedRtlModelThreads,
      std::move(*local),
      loom::evaluation::models::MappedRtlSimulatorBinding{
          mappedRtlProviderBuild}};
  if (llvm::Error error =
          loom::system_run::validateMappedRtlProviderOptions(options))
    return std::move(error);
  return options;
}

std::string child(llvm::StringRef parent, llvm::StringRef name) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, name);
  return path.str().str();
}

llvm::Expected<std::string> canonicalPath(llvm::StringRef source,
                                          bool requireDirectory) {
  std::error_code error;
  std::filesystem::path path =
      std::filesystem::weakly_canonical(source.str(), error);
  if (error || !path.is_absolute())
    return invalid("cannot canonicalize '" + source + "'");
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      (requireDirectory && !std::filesystem::is_directory(status)) ||
      (!requireDirectory && !std::filesystem::is_regular_file(status)))
    return invalid("path has the wrong filesystem kind: '" + source + "'");
  return path.string();
}

llvm::Expected<std::string> readText(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return ioError("cannot read '" + path +
                   "': " + buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

llvm::Error copyRegularDirectory(llvm::StringRef source,
                                 llvm::StringRef destination) {
  std::error_code error;
  if (!std::filesystem::create_directory(destination.str(), error) || error)
    return ioError("cannot create '" + destination + "'");
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(source.str(), error)) {
    if (error)
      return ioError("cannot enumerate '" + source + "'");
    const auto status = entry.symlink_status(error);
    if (error || !entry.is_regular_file(error) ||
        std::filesystem::is_symlink(status))
      return invalid("package directory contains a non-regular entry");
    const std::filesystem::path target =
        std::filesystem::path(destination.str()) / entry.path().filename();
    if (!std::filesystem::copy_file(
            entry.path(), target, std::filesystem::copy_options::none, error) ||
        error)
      return ioError("cannot copy package entry '" + entry.path().string() +
                     "'");
  }
  if (error)
    return ioError("cannot enumerate '" + source + "'");
  return llvm::Error::success();
}

struct InitializedWorkspace final {
  std::string path;
  loom::application::ImportedApplicationPackage package;
};

llvm::Expected<InitializedWorkspace> initializeWorkspace() {
  auto source = canonicalPath(deploymentPackage, true);
  if (!source)
    return source.takeError();
  auto sourcePackage = loom::application::importApplicationPackage(*source);
  if (!sourcePackage)
    return sourcePackage.takeError();
  std::filesystem::path output(outputWorkspace.getValue());
  std::error_code error;
  output = std::filesystem::absolute(output, error).lexically_normal();
  if (error || output.filename().empty())
    return invalid("cannot resolve output workspace");
  if (std::filesystem::exists(output, error) || error)
    return invalid("output workspace already exists");
  if (!std::filesystem::create_directory(output, error) || error)
    return ioError("cannot create output workspace");

  const std::filesystem::path sourceRoot =
      std::filesystem::path(*source) / "root";
  const auto rootStatus = std::filesystem::symlink_status(sourceRoot, error);
  if (error || !std::filesystem::is_regular_file(rootStatus) ||
      std::filesystem::is_symlink(rootStatus))
    return invalid("Deployment package root is not a regular file");
  if (!std::filesystem::copy_file(sourceRoot, output / "root",
                                  std::filesystem::copy_options::none, error) ||
      error)
    return ioError("cannot copy Deployment package root");
  const std::filesystem::path sourceApplication =
      std::filesystem::path(*source) / "application";
  const auto applicationStatus =
      std::filesystem::symlink_status(sourceApplication, error);
  if (error || !std::filesystem::is_regular_file(applicationStatus) ||
      std::filesystem::is_symlink(applicationStatus))
    return invalid("Application package root is not a regular file");
  if (!std::filesystem::copy_file(sourceApplication, output / "application",
                                  std::filesystem::copy_options::none, error) ||
      error)
    return ioError("cannot copy Application package root");
  if (llvm::Error copyError = copyRegularDirectory(
          child(*source, "objects"), (output / "objects").string()))
    return std::move(copyError);
  if (llvm::Error copyError = copyRegularDirectory(child(*source, "blobs"),
                                                   (output / "blobs").string()))
    return std::move(copyError);
  auto workspacePackage =
      loom::application::importApplicationPackage(output.string());
  if (!workspacePackage)
    return invalid("copied Application package failed independent import: " +
                   llvm::toString(workspacePackage.takeError()));
  return InitializedWorkspace{output.string(), std::move(*workspacePackage)};
}

struct Readiness final {
  loom::runtime::Gem5BuildIdentity identity;
  std::string path;
  std::string binary;
};

llvm::Expected<Readiness> readReadiness() {
  auto path = canonicalPath(gem5Readiness, false);
  if (!path)
    return path.takeError();
  auto contents = readText(*path);
  if (!contents)
    return contents.takeError();
  auto value = llvm::json::parse(*contents);
  if (!value)
    return value.takeError();
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return invalid("gem5 readiness is not an object");
  const auto schema = object->getString("schema");
  const auto repository = object->getString("gem5_repository_identity");
  const auto commit = object->getString("gem5_full_commit_identity");
  const auto configuration = object->getString("build_configuration_digest");
  const auto fingerprint = object->getString("binary_sha256");
  const auto binary = object->getString("binary");
  if (!schema || *schema != "loom.gem5_build_readiness.1" || !repository ||
      !commit || !configuration || !fingerprint || !binary)
    return invalid("gem5 readiness omits an identity field");
  auto binaryPath = canonicalPath(*binary, false);
  if (!binaryPath)
    return binaryPath.takeError();
  return Readiness{{repository->str(), commit->str(), configuration->str(),
                    fingerprint->str()},
                   std::move(*path),
                   std::move(*binaryPath)};
}

struct PublishedInputs final {
  loom::ArtifactRootReference workload;
  loom::ArtifactRootReference runtimeInput;
};

llvm::Expected<PublishedInputs>
loadInputs(const loom::application::ApplicationRuntimeManifest &manifest,
           const loom::deployment::FinalizedDeployment &deployment,
           const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  auto endpointInputs =
      loom::application::materializeApplicationEndpointActivationInputs(
          manifest, artifacts, blobs);
  if (!endpointInputs)
    return endpointInputs.takeError();
  const auto entry = llvm::find_if(*endpointInputs, [&](const auto &candidate) {
    return candidate.endpoint.mapping == manifest.selectedMapping() &&
           candidate.endpoint.deployment == deployment.reference();
  });
  if (entry == endpointInputs->end())
    return invalid("Application activation inputs omit the entry Deployment");
  auto imported = loom::sim::importSystemSimulationInputs(
      entry->inputs.workload, entry->inputs.runtimeInput, artifacts, blobs);
  if (!imported)
    return imported.takeError();
  if (imported->deployment.reference() != deployment.reference())
    return invalid("Application activation inputs name a foreign Deployment");
  return PublishedInputs{entry->inputs.workload, entry->inputs.runtimeInput};
}

llvm::Expected<std::shared_ptr<loom::runtime::FabricModelRuntimeProvider>>
createApplicationRuntimeProvider(
    const loom::application::FinalizedApplicationRuntimeManifest &manifest,
    const loom::deployment::FinalizedDeployment &entry,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  std::vector<loom::ArtifactRootReference> endpoints{entry.reference()};
  if (manifest.manifest().transitionGraph())
    for (const auto &endpoint :
         manifest.manifest().transitionGraph()->endpoints)
      if (endpoint.deployment &&
          !llvm::is_contained(endpoints, *endpoint.deployment))
        endpoints.push_back(*endpoint.deployment);

  std::vector<loom::ArtifactIdentity> implementations;
  for (const loom::ArtifactRootReference &reference : endpoints) {
    auto imported =
        loom::deployment::importDeployment(reference, artifacts, blobs);
    if (!imported)
      return imported.takeError();
    for (const loom::deployment::DeploymentHardwareBinding &binding :
         imported->deployment().hardwareBindings())
      if (!llvm::is_contained(implementations,
                              binding.hardwareImplementation.artifact))
        implementations.push_back(binding.hardwareImplementation.artifact);
  }
  if (implementations.empty())
    return invalid("resource-time application has no hardware implementation");
  return loom::runtime::createFabricModelRuntimeProvider(
      {{std::move(implementations)}});
}

/// The synchronous resource-time drive of one run: the loaded Application
/// Deployment with its prepared selector, the gem5 endpoint table derived from
/// the manifest graph, and the root events the controller acknowledged. The
/// controlled System DFG cell consults it at every root event before the
/// device continues; the trace is published from this session only.
struct ResourceTimeDrive final {
  loom::application::LoadedApplicationDeployment loaded;
  loom::runtime::Gem5RootEventEndpointTable endpoints;
  std::vector<loom::runtime::Gem5RootEventAcknowledgement> acknowledgements;
};

/// Typed refusal of the synchronous drive. The run still executes the entry
/// Deployment on both engines but publishes no activation evidence.
struct ResourceTimeDriveRefusal final {
  loom::runtime::Gem5RootEventControlErrorReason reason;
};

using ResourceTimeDriveOutcome =
    std::variant<std::monostate, ResourceTimeDrive, ResourceTimeDriveRefusal>;

llvm::Expected<ResourceTimeDriveOutcome> prepareResourceTimeDrive(
    const loom::application::FinalizedApplicationRuntimeManifest &manifest,
    const loom::deployment::FinalizedDeployment &deployment,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  if (!manifest.manifest().transitionGraph())
    return ResourceTimeDriveOutcome{std::monostate{}};
  auto endpoints = loom::runtime::deriveGem5RootEventEndpointTable(
      *manifest.manifest().transitionGraph(), artifacts);
  if (!endpoints) {
    std::optional<loom::runtime::Gem5RootEventControlErrorReason> refusal;
    llvm::Error error = llvm::handleErrors(
        endpoints.takeError(),
        [&](std::unique_ptr<loom::runtime::Gem5RootEventControlError> typed)
            -> llvm::Error {
          if (typed->reason() !=
              loom::runtime::Gem5RootEventControlErrorReason::NonTerminalEdge)
            return llvm::Error(std::move(typed));
          refusal = typed->reason();
          return llvm::Error::success();
        });
    if (error)
      return std::move(error);
    return ResourceTimeDriveOutcome{ResourceTimeDriveRefusal{*refusal}};
  }
  auto provider =
      createApplicationRuntimeProvider(manifest, deployment, artifacts, blobs);
  if (!provider)
    return provider.takeError();
  auto loaded = loom::application::loadApplicationDeployment(
      manifest, deployment, {*provider, 0}, artifacts, blobs);
  if (!loaded)
    return loaded.takeError();
  return ResourceTimeDriveOutcome{
      ResourceTimeDrive{std::move(*loaded), std::move(*endpoints), {}}};
}

/// The controlled cell's device-published lifecycle must be exactly the
/// acknowledged sequence; a divergence means the device continued past an
/// unacknowledged event.
llvm::Error validateAcknowledgedRootLifecycle(
    llvm::ArrayRef<loom::runtime::Gem5RootEventAcknowledgement> acknowledged,
    llvm::ArrayRef<loom::sim::SystemRootLifecycleObservation> published) {
  if (acknowledged.size() != published.size())
    return invalid("controlled System cell published a root lifecycle that "
                   "differs from the acknowledged events");
  for (std::size_t index = 0; index != published.size(); ++index) {
    const loom::sim::SystemRootLifecycleObservation &expected =
        acknowledged[index].observation;
    const loom::sim::SystemRootLifecycleObservation &observed =
        published[index];
    if (expected.event != observed.event ||
        expected.occurrence != observed.occurrence ||
        expected.coordinate.gem5Tick != observed.coordinate.gem5Tick ||
        expected.coordinate.delta != observed.coordinate.delta)
      return invalid("controlled System cell root event differs from its "
                     "acknowledged decision");
  }
  return llvm::Error::success();
}

llvm::Expected<loom::application::FinalizedApplicationResourceTimeExecutionTrace>
publishResourceTimeDriveTrace(ResourceTimeDrive &drive,
                              const loom::ArtifactStore &artifacts,
                              const loom::BlobStore &blobs) {
  if (llvm::Error error =
          drive.loaded.resourceTimeExecution()->joinMappedRoots())
    return std::move(error);
  return drive.loaded.publishResourceTimeExecutionTrace(artifacts, blobs);
}

llvm::Expected<loom::evaluation::CaseArtifactResolution>
buildResolution(const loom::deployment::FinalizedDeployment &deployment,
                const loom::runtime::FinalizedGem5SimulationBinding &binding,
                const PublishedInputs &inputs,
                const loom::ArtifactStore &artifacts,
                const loom::BlobStore &blobs) {
  auto package = loom::deployment::deriveDeploymentPackageClosure(
      deployment, artifacts, blobs);
  if (!package)
    return package.takeError();
  std::map<loom::ArtifactRootReference,
           std::vector<loom::ArtifactRootReference>,
           decltype(&loom::artifactRootReferenceLess)>
      entries(&loom::artifactRootReferenceLess);
  for (const loom::ArtifactRootReference &root : package->artifacts())
    entries.emplace(root, std::vector<loom::ArtifactRootReference>{});
  std::vector<loom::ArtifactRootReference> deploymentClosure;
  for (const loom::ArtifactRootReference &root : package->artifacts())
    if (root != deployment.reference())
      deploymentClosure.push_back(root);
  entries[deployment.reference()] = deploymentClosure;
  entries[inputs.workload] = package->artifacts().vec();
  std::vector<loom::ArtifactRootReference> runtimeClosure =
      package->artifacts().vec();
  runtimeClosure.push_back(inputs.workload);
  entries[inputs.runtimeInput] = std::move(runtimeClosure);

  std::set<loom::ArtifactRootReference,
           decltype(&loom::artifactRootReferenceLess)>
      fabricClosure(&loom::artifactRootReferenceLess);
  std::function<llvm::Error(const loom::ArtifactRootReference &)> addFabric =
      [&](const loom::ArtifactRootReference &root) -> llvm::Error {
    if (!fabricClosure.insert(root).second)
      return llvm::Error::success();
    auto imported = loom::fabric::importEntireFabricRoot(root, artifacts);
    if (!imported)
      return imported.takeError();
    entries.emplace(root, std::vector<loom::ArtifactRootReference>{});
    for (const loom::fabric::FabricDirectDependency &dependency :
         imported->directDependencies())
      if (llvm::Error error = addFabric(dependency.root))
        return error;
    return llvm::Error::success();
  };
  if (llvm::Error error = addFabric(binding.binding().fabric()))
    return std::move(error);
  if (llvm::Error error =
          addFabric(binding.binding().interconnectImplementation()))
    return std::move(error);
  entries[binding.reference()] = {fabricClosure.begin(), fabricClosure.end()};

  std::vector<loom::evaluation::CaseArtifactResolution::Entry> result;
  result.reserve(entries.size());
  for (auto &[root, closure] : entries)
    result.push_back({root, std::move(closure)});
  return loom::evaluation::CaseArtifactResolution::get(std::move(result));
}

enum class Engine : std::uint8_t { Dfg, Cgra, Rtl };

llvm::StringRef engineSpelling(Engine engine) {
  switch (engine) {
  case Engine::Dfg:
    return "dfg";
  case Engine::Cgra:
    return "cgra";
  case Engine::Rtl:
    return "rtl";
  }
  llvm_unreachable("unknown System execution engine");
}

llvm::StringRef spatialEngineBundleName(Engine engine) {
  switch (engine) {
  case Engine::Dfg:
    return "spatial-dfg";
  case Engine::Cgra:
    return "spatial-cgra";
  case Engine::Rtl:
    return "spatial-rtl";
  }
  llvm_unreachable("unknown Spatial execution engine");
}

using loom::system_run::ObservedSpatialInvocation;

struct CompletedRun final {
  Engine engine;
  loom::ArtifactRootReference request;
  loom::ArtifactRootReference evidence;
  loom::ArtifactRootReference execution;
  loom::sim::CanonicalSimulationExecution importedExecution;
  std::vector<ObservedSpatialInvocation> spatialInvocations;
  std::optional<loom::ArtifactRootReference> productOracleRequest;
  std::optional<loom::ArtifactRootReference> productOracleEvidence;
};

llvm::Expected<std::vector<ObservedSpatialInvocation>> readSpatialInvocations(
    const loom::external_tool::PreparedExternalToolInvocation &prepared) {
  auto projectionText = readText(
      child(prepared.bundleRoot, "drivers/gem5-system-projection.json"));
  if (!projectionText)
    return projectionText.takeError();
  auto projection = llvm::json::parse(*projectionText);
  if (!projection)
    return projection.takeError();
  const llvm::json::Object *object = projection->getAsObject();
  const llvm::json::Array *bridges =
      object ? object->getArray("bridges") : nullptr;
  const llvm::json::Object *dispatch =
      object ? object->getObject("dispatch") : nullptr;
  const llvm::json::Array *dispatchTargets =
      dispatch ? dispatch->getArray("targets") : nullptr;
  const auto schema = object ? object->getString("schema") : std::nullopt;
  if (!schema || *schema != "loom.gem5_system_projection.13" || !bridges ||
      bridges->empty() || !dispatchTargets || dispatchTargets->empty())
    return invalid("gem5 projection contains no Spatial bridge");

  std::vector<ObservedSpatialInvocation> invocations;
  invocations.reserve(dispatchTargets->size());
  std::vector<bool> claimedTargets(dispatchTargets->size(), false);
  for (const auto indexed : llvm::enumerate(*bridges)) {
    const llvm::json::Object *bridge = indexed.value().getAsObject();
    if (!bridge)
      return invalid("gem5 projection contains a non-object bridge");
    const llvm::json::Array *dispatchTargetOrdinals =
        bridge->getArray("dispatch_target_ordinals");
    const auto accCoreReference = bridge->getString("acc_core_ref");
    const llvm::json::Array *executionContextKeys =
        bridge->getArray("execution_context_keys");
    const llvm::json::Array *spatialWorkloads =
        bridge->getArray("spatial_workloads");
    if (!dispatchTargetOrdinals || dispatchTargetOrdinals->empty() ||
        !accCoreReference || accCoreReference->empty() ||
        !executionContextKeys || !spatialWorkloads ||
        executionContextKeys->size() != dispatchTargetOrdinals->size() ||
        spatialWorkloads->size() != dispatchTargetOrdinals->size())
      return invalid("gem5 bridge has no canonical execution target identity");
    std::vector<std::uint64_t> targetOrdinals;
    std::vector<std::string> contextKeys;
    targetOrdinals.reserve(dispatchTargetOrdinals->size());
    contextKeys.reserve(executionContextKeys->size());
    for (std::size_t ordinal = 0; ordinal != dispatchTargetOrdinals->size();
         ++ordinal) {
      const auto target = (*dispatchTargetOrdinals)[ordinal].getAsInteger();
      const auto context = (*executionContextKeys)[ordinal].getAsString();
      if (!target || *target < 0 ||
          static_cast<std::uint64_t>(*target) >= dispatchTargets->size() ||
          claimedTargets[static_cast<std::size_t>(*target)] || !context ||
          context->empty() ||
          (!targetOrdinals.empty() &&
           targetOrdinals.back() >= static_cast<std::uint64_t>(*target)))
        return invalid("gem5 bridge target table is not canonical");
      claimedTargets[static_cast<std::size_t>(*target)] = true;
      targetOrdinals.push_back(static_cast<std::uint64_t>(*target));
      contextKeys.push_back(context->str());
    }
    const auto resultPath = bridge->getString("result_path");
    const std::string expectedPath =
        "outputs/spatial-bridge-" + std::to_string(indexed.index()) + ".result";
    if (!resultPath || *resultPath != expectedPath)
      return invalid("gem5 projection has a noncanonical bridge result path");
    std::vector<loom::ArtifactRootReference> workloadReferences;
    workloadReferences.reserve(spatialWorkloads->size());
    for (const llvm::json::Value &workload : *spatialWorkloads) {
      const auto workloadIdentityText = workload.getAsString();
      if (!workloadIdentityText)
        return invalid("gem5 bridge workload identity is not a string");
      auto workloadIdentity =
          loom::parseArtifactIdentityHex(*workloadIdentityText);
      if (!workloadIdentity)
        return workloadIdentity.takeError();
      workloadReferences.push_back(
          {loom::sim::simulationWorkloadSchema.identity.str(),
           loom::sim::simulationWorkloadSchema.version, *workloadIdentity});
    }
    if (workloadReferences.size() != targetOrdinals.size())
      return invalid("gem5 bridge omits its exact Spatial workload");
    auto resultText = readText(child(prepared.bundleRoot, expectedPath));
    if (!resultText)
      return resultText.takeError();
    const std::vector<std::uint8_t> resultBytes(resultText->begin(),
                                                resultText->end());
    loom::runtime::Gem5BridgeResultCollection bridgeResults;
    std::string diagnostic;
    if (!loom::runtime::decodeGem5BridgeResultCollection(
            resultBytes, bridgeResults, diagnostic))
      return invalid("cannot decode verified gem5 bridge result: " +
                     llvm::Twine(diagnostic));
    if (bridgeResults.results.empty())
      return invalid("gem5 bridge did not execute a declared target");
    std::vector<bool> observedSessionEntries(targetOrdinals.size());
    for (const auto resultIndexed : llvm::enumerate(bridgeResults.results)) {
      const loom::runtime::Gem5BridgeResult &bridgeResult =
          resultIndexed.value();
      if (bridgeResult.status != 0 ||
          bridgeResult.sequence != resultIndexed.index())
        return invalid("gem5 bridge invocation sequence did not retire");
      loom::runtime::SpatialInvocationResultWire invocationResult;
      if (!loom::runtime::decodeSpatialInvocationResultWire(
              bridgeResult.result, invocationResult, diagnostic))
        return invalid("cannot decode verified Spatial invocation result: " +
                       llvm::Twine(diagnostic));
      if (invocationResult.sessionEntryOrdinal >= targetOrdinals.size())
        return invalid("gem5 bridge result names an absent target entry");
      const std::size_t sessionEntryOrdinal =
          static_cast<std::size_t>(invocationResult.sessionEntryOrdinal);
      observedSessionEntries[sessionEntryOrdinal] = true;
      if (invocationResult.invocation.empty() || !invocationResult.runtimeInput)
        return invalid("public execution matrix requires a complete dynamic "
                       "invocation");
      invocations.push_back(
          {targetOrdinals[sessionEntryOrdinal], accCoreReference->str(),
           contextKeys[sessionEntryOrdinal],
           workloadReferences[sessionEntryOrdinal],
           std::move(invocationResult.invocation),
           std::move(*invocationResult.runtimeInput),
           std::move(invocationResult.spatialBoundaryResult)});
    }
    if (llvm::is_contained(observedSessionEntries, false))
      return invalid("gem5 bridge results omit a declared target entry");
  }
  if (llvm::is_contained(claimedTargets, false))
    return invalid("gem5 bridge sessions omit a dispatch target");
  return invocations;
}

/// Executes one System cell. A non-null `drive` makes the cell the
/// completion-controlled invocation: the host controller acknowledges every
/// root event through the prepared selector before the device continues, the
/// bundle runs fresh, and the published lifecycle must equal the acknowledged
/// sequence.
llvm::Expected<CompletedRun>
execute(Engine engine, llvm::StringRef workspace,
        const loom::deployment::FinalizedDeployment &deployment,
        const loom::runtime::FinalizedGem5SimulationBinding &binding,
        const PublishedInputs &inputs,
        const loom::evaluation::CaseArtifactResolution &resolution,
        const Readiness &readiness, const loom::ArtifactStore &artifacts,
        const loom::BlobStore &blobs, ResourceTimeDrive *drive) {
  auto subjects = loom::evaluation::EvaluationSubjectBindings::get(
      {{loom::evaluation::CaseSubjectRoleRef(0), {deployment.reference()}},
       {loom::evaluation::CaseSubjectRoleRef(1), {binding.reference()}}});
  if (!subjects)
    return subjects.takeError();
  auto evaluationCase = loom::evaluation::EvaluationCase::get(
      loom::evaluation::systemSimulationCaseSignatureRef(),
      std::move(*subjects), inputs.workload, inputs.runtimeInput, {},
      resolution, artifacts, blobs);
  if (!evaluationCase)
    return evaluationCase.takeError();
  const auto modelKind =
      engine == Engine::Dfg
          ? loom::evaluation::BuiltinEvaluationModel::Gem5SystemDfg
          : loom::evaluation::BuiltinEvaluationModel::Gem5SystemCgra;
  auto descriptor =
      loom::evaluation::builtinEvaluationModelDescriptorRef(modelKind);
  if (!descriptor)
    return descriptor.takeError();
  auto model = loom::evaluation::ResolvedModelBinding::project(
      *descriptor, {}, loom::defaultResolvedConfig());
  if (!model)
    return model.takeError();
  auto request = loom::evaluation::EvaluationRequest::get(
      *evaluationCase, {}, {}, std::move(*model), 0, resolution, artifacts,
      blobs);
  if (!request)
    return request.takeError();
  auto requestRef =
      loom::evaluation::publishEvaluationRequest(*request, artifacts);
  if (!requestRef)
    return requestRef.takeError();

  loom::external_tool::LocalToolConfig local;
  local.runtimePolicy = loom::external_tool::RuntimePolicy::Host;
  auto &gem5 = local.tools[loom::external_tool::gem5Provider().binding.key];
  gem5.binding.executable = readiness.binary;
  gem5.providerOptions["readiness"] = readiness.path;
  const llvm::StringRef engineName =
      engine == Engine::Dfg ? "system-dfg" : "system-cgra";
  const loom::external_tool::ExternalToolPreparationContext preparationContext{
      std::move(local), child(child(workspace, "bundles"), engineName)};
  std::optional<loom::evaluation::EvaluationModelPreparation> preparation;
  if (drive) {
    auto controlled =
        loom::runtime::prepareGem5SystemCompletionControlledInvocation(
            *request, resolution, artifacts, blobs, preparationContext,
            drive->endpoints);
    if (!controlled)
      return controlled.takeError();
    auto *external =
        std::get_if<loom::external_tool::PreparedExternalToolInvocation>(
            &*controlled);
    if (!external)
      return invalid(engineName + " is unsupported for the exact Deployment");
    auto bound = loom::evaluation::bindPreparedEvaluationModelInvocation(
        *request, resolution, *external, artifacts, blobs);
    if (!bound)
      return bound.takeError();
    preparation.emplace(std::move(*bound));
  } else {
    auto ordinary = loom::evaluation::prepareEvaluationModelInvocation(
        *request, resolution, artifacts, blobs, preparationContext);
    if (!ordinary)
      return ordinary.takeError();
    preparation.emplace(std::move(*ordinary));
  }
  const auto *prepared =
      std::get_if<loom::evaluation::EvaluationModelPreparedInvocation>(
          &*preparation);
  if (!prepared)
    return invalid(engineName + " is unsupported for the exact Deployment");
  const loom::external_tool::PreparedExternalToolInvocation &external =
      prepared->externalInvocation();
  std::optional<loom::runtime::Gem5RootEventController> controller;
  if (drive) {
    auto listening = loom::runtime::Gem5RootEventController::listen(
        external.bundleRoot, drive->endpoints.dataflow,
        [drive](const loom::sim::SystemRootLifecycleObservation &observation) {
          return drive->loaded.driveGem5RootEvent(observation,
                                                  drive->endpoints);
        });
    if (!listening)
      return listening.takeError();
    controller.emplace(std::move(*listening));
  }
  auto execution =
      loom::external_tool::executeExternalToolInvocationBundleObserved(
          external, {},
          drive ? loom::external_tool::ExternalToolResultReusePolicy::
                      RequireFresh
                : loom::external_tool::ExternalToolResultReusePolicy::
                      AllowExactReuse);
  if (controller) {
    auto acknowledged = controller->finish();
    if (!acknowledged) {
      if (!execution)
        llvm::consumeError(execution.takeError());
      return acknowledged.takeError();
    }
    drive->acknowledgements = std::move(*acknowledged);
  }
  if (!execution)
    return execution.takeError();
  if (execution->exitCode != 0)
    return invalid(engineName + " external invocation exited with status " +
                   llvm::Twine(execution->exitCode));
  auto evidence = loom::evaluation::importEvaluationModelInvocation(
      *request, resolution, *prepared, *execution, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  if (evidence->outcomeKind() !=
      loom::evaluation::EvidenceOutcomeKind::Completed)
    return invalid(engineName + " published " +
                   loom::evaluation::toString(evidence->outcomeKind()) +
                   " Evidence");
  auto evidenceRef =
      loom::evaluation::publishEvaluationEvidence(*evidence, artifacts);
  if (!evidenceRef)
    return evidenceRef.takeError();
  if (evidence->outputBindings().size() != 1 ||
      evidence->outputBindings().front().artifacts.size() != 1)
    return invalid(engineName + " did not publish one execution");
  const loom::ArtifactRootReference executionRef =
      evidence->outputBindings().front().artifacts.front();
  auto imported = loom::sim::importSimulationExecution(executionRef, resolution,
                                                       artifacts, blobs);
  if (!imported)
    return imported.takeError();
  if (!std::holds_alternative<loom::sim::RetiredExecution>(
          imported->terminal()) ||
      !imported->system())
    return invalid(engineName + " did not retire a System execution");
  if (drive)
    if (llvm::Error error = validateAcknowledgedRootLifecycle(
            drive->acknowledgements,
            imported->system()->progressObservations.rootLifecycle))
      return error;
  auto spatialInvocations = readSpatialInvocations(external);
  if (!spatialInvocations)
    return spatialInvocations.takeError();
  return CompletedRun{
      engine,       std::move(*requestRef), std::move(*evidenceRef),
      executionRef, std::move(*imported),   std::move(*spatialInvocations)};
}

using loom::system_run::SpatialInvocationCase;

struct CompletedSpatialRun final {
  Engine engine;
  std::size_t invocationOrdinal = 0;
  loom::ArtifactRootReference request;
  loom::ArtifactRootReference evidence;
  loom::ArtifactRootReference execution;
  loom::sim::CanonicalSimulationExecution importedExecution;
};

struct SpatialCgraProfileSample final {
  std::uint64_t attemptOrdinal = 0;
  loom::ArtifactRootReference request;
  loom::ArtifactRootReference evidence;
  loom::ArtifactRootReference execution;
  std::uint64_t referenceCycles = 0;
  loom::evaluation::models::CgraSimulationAttemptProfile attempt;
};

struct SpatialCgraProfileRecord final {
  std::size_t invocationOrdinal = 0;
  std::vector<SpatialCgraProfileSample> warmups;
  std::vector<SpatialCgraProfileSample> measurements;
};

struct ProfiledSpatialRun final {
  CompletedSpatialRun completed;
  SpatialCgraProfileSample sample;
};

llvm::Expected<CompletedSpatialRun>
completeSpatialRun(Engine engine, std::size_t invocationOrdinal,
                   const loom::evaluation::EvaluationRequest &request,
                   const loom::evaluation::CaseArtifactResolution &resolution,
                   loom::evaluation::EvaluationEvidence evidence,
                   const loom::ArtifactStore &artifacts,
                   const loom::BlobStore &blobs) {
  const llvm::StringRef engineName = spatialEngineBundleName(engine);
  if (evidence.outcomeKind() !=
      loom::evaluation::EvidenceOutcomeKind::Completed)
    return invalid(engineName + " published " +
                   loom::evaluation::toString(evidence.outcomeKind()) +
                   " Evidence");
  auto requestReference =
      loom::evaluation::publishEvaluationRequest(request, artifacts);
  if (!requestReference)
    return requestReference.takeError();
  auto evidenceReference =
      loom::evaluation::publishEvaluationEvidence(evidence, artifacts);
  if (!evidenceReference)
    return evidenceReference.takeError();
  if (evidence.outputBindings().size() != 1 ||
      evidence.outputBindings().front().artifacts.size() != 1)
    return invalid(engineName + " did not publish one execution");
  const loom::ArtifactRootReference executionReference =
      evidence.outputBindings().front().artifacts.front();
  auto imported = loom::sim::importSimulationExecution(
      executionReference, resolution, artifacts, blobs);
  if (!imported)
    return imported.takeError();
  if (!std::holds_alternative<loom::sim::RetiredExecution>(
          imported->terminal()) ||
      !imported->spatial())
    return invalid(engineName + " did not retire a Spatial execution");
  return CompletedSpatialRun{engine,
                             invocationOrdinal,
                             std::move(*requestReference),
                             std::move(*evidenceReference),
                             executionReference,
                             std::move(*imported)};
}

llvm::Expected<loom::evaluation::models::PreparedCgraSimulationEvaluation>
prepareSpatialCgra(const SpatialInvocationCase &invocation,
                   const loom::ArtifactStore &artifacts,
                   const loom::BlobStore &blobs) {
  return loom::evaluation::models::prepareCgraSimulationEvaluation(
      invocation.dataflow, invocation.fabric, invocation.spatialMapping,
      invocation.workload, invocation.runtimeInput,
      loom::defaultResolvedConfig(), artifacts, blobs);
}

llvm::Expected<CompletedSpatialRun>
executeSpatial(Engine engine, const SpatialInvocationCase &invocation,
               const loom::ArtifactStore &artifacts,
               const loom::BlobStore &blobs) {
  if (engine == Engine::Dfg) {
    auto prepared = loom::evaluation::models::prepareDfgSimulationEvaluation(
        invocation.dataflow, invocation.workload, invocation.runtimeInput,
        loom::defaultResolvedConfig(), artifacts, blobs);
    if (!prepared)
      return prepared.takeError();
    auto evidence = loom::evaluation::models::evaluateDfgSimulation(
        *prepared, {1000000, std::nullopt}, artifacts, blobs);
    if (!evidence)
      return evidence.takeError();
    return completeSpatialRun(engine, invocation.ordinal, prepared->request,
                              prepared->resolution, std::move(*evidence),
                              artifacts, blobs);
  }
  if (engine != Engine::Cgra)
    return invalid("in-process Spatial execution selected an external engine");
  auto prepared = prepareSpatialCgra(invocation, artifacts, blobs);
  if (!prepared)
    return prepared.takeError();
  auto evidence = loom::evaluation::models::evaluateCgraSimulation(
      *prepared, {1000000, std::nullopt}, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  return completeSpatialRun(engine, invocation.ordinal, prepared->request,
                            prepared->resolution, std::move(*evidence),
                            artifacts, blobs);
}

llvm::Expected<ProfiledSpatialRun> executeProfiledSpatialCgra(
    const loom::evaluation::models::PreparedCgraSimulationEvaluation &prepared,
    const SpatialInvocationCase &invocation, std::uint64_t attemptOrdinal,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  auto evaluated =
      loom::evaluation::models::evaluateCgraSimulationWithAttemptProfile(
          prepared, {1000000, std::nullopt}, artifacts, blobs);
  if (!evaluated)
    return evaluated.takeError();
  auto completed = completeSpatialRun(
      Engine::Cgra, invocation.ordinal, prepared.request, prepared.resolution,
      std::move(evaluated->evidence), artifacts, blobs);
  if (!completed)
    return completed.takeError();
  const auto *spatial = completed->importedExecution.spatial();
  if (!spatial || !spatial->progressObservations.graphRetirementVisible ||
      spatial->progressObservations.graphRetirementVisible->referenceCycle
              .denominator() != 1)
    return invalid("profiled Spatial CGRA execution has no integral "
                   "retirement cycle");
  const std::uint64_t referenceCycles =
      spatial->progressObservations.graphRetirementVisible->referenceCycle
          .numerator();
  if (!evaluated->attemptProfile || referenceCycles == 0 ||
      evaluated->attemptProfile->engineActiveWallNanoseconds == 0 ||
      evaluated->attemptProfile->counters.eventFrameCount == 0)
    return invalid("profiled Spatial CGRA execution contains no active work");
  SpatialCgraProfileSample sample{
      attemptOrdinal,      completed->request,
      completed->evidence, completed->execution,
      referenceCycles,     std::move(*evaluated->attemptProfile)};
  return ProfiledSpatialRun{std::move(*completed), std::move(sample)};
}

void writeOptionalUnsigned(llvm::json::OStream &json, llvm::StringRef key,
                           std::optional<std::uint64_t> value) {
  if (value)
    json.attribute(key, *value);
  else
    json.attribute(key, nullptr);
}

void writeProfileSample(llvm::json::OStream &json,
                        const SpatialCgraProfileSample &sample) {
  const auto &profile = sample.attempt;
  const auto &counters = profile.counters;
  json.object([&] {
    json.attribute("attempt_ordinal", sample.attemptOrdinal);
    json.attribute("reference_cycles", sample.referenceCycles);
    json.attribute("attempt_setup_wall_nanoseconds",
                   profile.inputLoadWallNanoseconds);
    writeOptionalUnsigned(json, "attempt_setup_process_cpu_nanoseconds",
                          profile.inputLoadCpuNanoseconds);
    json.attribute("engine_active_wall_nanoseconds",
                   profile.engineActiveWallNanoseconds);
    writeOptionalUnsigned(json, "engine_active_process_cpu_nanoseconds",
                          profile.engineActiveCpuNanoseconds);
    json.attribute("observation_projection_wall_nanoseconds",
                   profile.observationProjectionWallNanoseconds);
    writeOptionalUnsigned(json,
                          "observation_projection_process_cpu_nanoseconds",
                          profile.observationProjectionCpuNanoseconds);
    json.attribute("artifact_publication_wall_nanoseconds",
                   profile.artifactPublicationWallNanoseconds);
    writeOptionalUnsigned(json, "artifact_publication_process_cpu_nanoseconds",
                          profile.artifactPublicationCpuNanoseconds);
    json.attribute("event_frame_count", counters.eventFrameCount);
    json.attribute("physical_request_count", counters.physicalRequestCount);
    json.attribute("physical_grant_count", counters.physicalGrantCount);
    json.attribute("physical_retirement_count",
                   counters.physicalRetirementCount);
    json.attribute("physical_grant_wait_cycle_sum",
                   counters.physicalGrantWaitCycleSum);
    json.attribute("physical_grant_wait_cycle_max",
                   counters.physicalGrantWaitCycleMax);
    json.attribute("physical_grant_delayed_count",
                   counters.physicalGrantDelayedCount);
    const auto writeRoot = [&](llvm::StringRef key,
                               const loom::ArtifactRootReference &root) {
      json.attributeObject(
          key, [&] { loom::writeArtifactRootReferenceJsonFields(json, root); });
    };
    writeRoot("request", sample.request);
    writeRoot("evidence", sample.evidence);
    writeRoot("execution", sample.execution);
  });
}

llvm::Expected<std::uint64_t> processPeakResidentBytes() {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss <= 0)
    return ioError("cannot read process peak resident memory");
  constexpr std::uint64_t bytesPerKibibyte = 1024;
  const std::uint64_t kibibytes = static_cast<std::uint64_t>(usage.ru_maxrss);
  if (kibibytes > std::numeric_limits<std::uint64_t>::max() / bytesPerKibibyte)
    return invalid("process peak resident memory exceeds the report domain");
  return kibibytes * bytesPerKibibyte;
}

llvm::Error writeSpatialCgraProfile(
    llvm::StringRef path,
    const loom::application::FinalizedApplicationRuntimeManifest &manifest,
    const loom::deployment::FinalizedDeployment &deployment,
    llvm::ArrayRef<SpatialInvocationCase> invocations,
    llvm::ArrayRef<SpatialCgraProfileRecord> profiles,
    std::uint64_t peakResidentBytes) {
  if (profiles.size() != invocations.size())
    return invalid("Spatial CGRA profile lost an invocation");
  std::string body;
  llvm::raw_string_ostream stream(body);
  llvm::json::OStream json(stream, 2);
  json.object([&] {
    json.attribute("schema", "loom.spatial_cgra_qualification.1");
    json.attribute("warmup_runs", spatialCgraWarmupRuns.getValue());
    json.attribute("measurement_runs", spatialCgraMeasurementRuns.getValue());
    json.attribute("process_peak_resident_bytes", peakResidentBytes);
    json.attributeObject("runtime_manifest", [&] {
      loom::writeArtifactRootReferenceJsonFields(json, manifest.reference());
    });
    json.attributeObject("deployment", [&] {
      loom::writeArtifactRootReferenceJsonFields(json, deployment.reference());
    });
    json.attributeArray("invocations", [&] {
      for (const auto indexed : llvm::enumerate(profiles)) {
        const SpatialInvocationCase &invocation = invocations[indexed.index()];
        const SpatialCgraProfileRecord &profile = indexed.value();
        json.object([&] {
          json.attribute("invocation_ordinal", profile.invocationOrdinal);
          json.attributeArray("dense_coordinates", [&] {
            for (std::uint64_t coordinate : invocation.denseCoordinates)
              json.value(coordinate);
          });
          const auto writeRoot = [&](llvm::StringRef key,
                                     const loom::ArtifactRootReference &root) {
            json.attributeObject(key, [&] {
              loom::writeArtifactRootReferenceJsonFields(json, root);
            });
          };
          writeRoot("dataflow", invocation.dataflow);
          writeRoot("spatial_mapping", invocation.spatialMapping);
          writeRoot("workload", invocation.workload);
          writeRoot("runtime_input", invocation.runtimeInput);
          json.attributeArray("warmups", [&] {
            for (const SpatialCgraProfileSample &sample : profile.warmups)
              writeProfileSample(json, sample);
          });
          json.attributeArray("measurements", [&] {
            for (const SpatialCgraProfileSample &sample : profile.measurements)
              writeProfileSample(json, sample);
          });
        });
      }
    });
  });
  llvm::SmallString<256> temporaryModel(path);
  temporaryModel.append(".tmp-%%%%%%");
  auto temporary = llvm::sys::fs::TempFile::create(
      temporaryModel, llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
  if (!temporary)
    return ioError("cannot create temporary Spatial CGRA profile: " +
                   llvm::toString(temporary.takeError()));
  llvm::scope_exit discardTemporary(
      [&] { llvm::consumeError(temporary->discard()); });
  {
    llvm::raw_fd_ostream output(temporary->FD, false);
    output << body << '\n';
    output.flush();
    if (std::error_code error = output.error()) {
      output.clear_error();
      return ioError("cannot write temporary Spatial CGRA profile: " +
                     error.message());
    }
  }
  const int temporaryFile = temporary->FD;
  temporary->FD = -1;
  if (std::error_code error =
          llvm::sys::Process::SafelyCloseFileDescriptor(temporaryFile))
    return ioError("cannot close temporary Spatial CGRA profile: " +
                   error.message());
  if (std::error_code error =
          llvm::sys::fs::create_hard_link(temporary->TmpName, path)) {
    if (error == std::errc::file_exists)
      return invalid("Spatial CGRA profile output already exists");
    return ioError("cannot publish Spatial CGRA profile: " + error.message());
  }
  discardTemporary.release();
  llvm::consumeError(temporary->discard());
  return llvm::Error::success();
}

bool haveEquivalentRootLifecycle(
    llvm::ArrayRef<loom::sim::SystemRootLifecycleObservation> lhs,
    llvm::ArrayRef<loom::sim::SystemRootLifecycleObservation> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  std::map<std::uint64_t, std::uint64_t> lhsOccurrences;
  std::map<std::uint64_t, std::uint64_t> rhsOccurrences;
  const auto normalized = [](auto &occurrences, std::uint64_t occurrence) {
    const std::uint64_t next = occurrences.size() + 1;
    return occurrences.try_emplace(occurrence, next).first->second;
  };
  for (std::size_t index = 0; index != lhs.size(); ++index)
    if (lhs[index].event != rhs[index].event ||
        normalized(lhsOccurrences, lhs[index].occurrence) !=
            normalized(rhsOccurrences, rhs[index].occurrence))
      return false;
  return true;
}

llvm::Error validateSystemResults(const CompletedRun &dfg,
                                  const CompletedRun &cgra) {
  const loom::sim::SystemSimulationExecution &dfgSystem =
      *dfg.importedExecution.system();
  const loom::sim::SystemSimulationExecution &cgraSystem =
      *cgra.importedExecution.system();
  const auto &lhs = dfgSystem.functionalObservations;
  const auto &rhs = cgraSystem.functionalObservations;
  if (!loom::sim::haveExactlyEqualSystemFunctionalObservations(lhs, rhs) ||
      !lhs.externalValueOutputs.empty() || !rhs.externalValueOutputs.empty() ||
      !lhs.externalStreamOutputs.empty() ||
      !rhs.externalStreamOutputs.empty())
    return invalid("System DFG and CGRA functional observations differ");
  if (!haveEquivalentRootLifecycle(
          dfgSystem.progressObservations.rootLifecycle,
          cgraSystem.progressObservations.rootLifecycle))
    return invalid("System DFG and CGRA root lifecycle observations differ");
  if (expectedI32.getNumOccurrences() == 0)
    return llvm::Error::success();
  if (lhs.valueResults.size() != 1)
    return invalid("expected-i32 requires exactly one value result");
  const auto *published =
      std::get_if<loom::sim::PublishedValueResult>(&lhs.valueResults.front());
  if (!published || published->value.tokenCount != 1 ||
      published->value.lanes.size() != 1 ||
      published->value.lanes.front().state !=
          loom::sim::SemanticState::Defined ||
      published->value.lanes.front().bits.getBitWidth() != 32)
    return invalid("observed result is not one defined i32");
  if (published->value.lanes.front().bits.getSExtValue() != expectedI32)
    return invalid("System result differs from the independent oracle");
  return llvm::Error::success();
}

llvm::Error publishProductOracleEvidence(
    const loom::application::FinalizedApplicationRuntimeManifest &manifest,
    CompletedRun &run,
    const loom::evaluation::CaseArtifactResolution &executionResolution,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  auto prepared = loom::application::prepareProductOracleEvaluation(
      manifest, run.execution, executionResolution,
      loom::defaultResolvedConfig(), artifacts, blobs);
  if (!prepared)
    return prepared.takeError();
  auto evidence = loom::application::evaluateProductOracle(
      *prepared, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  const auto *completed =
      std::get_if<loom::evaluation::CompletedEvidence>(&evidence->outcome());
  if (!completed || completed->findingResults.size() != 1 ||
      !std::holds_alternative<loom::evaluation::AbsentFinding>(
          completed->findingResults.front().result))
    return invalid(engineSpelling(run.engine) +
                   " System execution differs from the product oracle");
  auto request = loom::evaluation::publishEvaluationRequest(
      prepared->request, artifacts);
  if (!request)
    return request.takeError();
  auto published =
      loom::evaluation::publishEvaluationEvidence(*evidence, artifacts);
  if (!published)
    return published.takeError();
  run.productOracleRequest = std::move(*request);
  run.productOracleEvidence = std::move(*published);
  return llvm::Error::success();
}

llvm::Error validateSpatialResults(const SpatialInvocationCase &invocation,
                                   const CompletedSpatialRun &dfg,
                                   const CompletedSpatialRun &candidate) {
  if (dfg.invocationOrdinal != invocation.ordinal ||
      candidate.invocationOrdinal != invocation.ordinal ||
      dfg.engine != Engine::Dfg || candidate.engine == Engine::Dfg)
    return invalid("Spatial execution is attached to the wrong matrix cell");
  const auto &dfgObservations =
      dfg.importedExecution.spatial()->functionalObservations;
  const auto &candidateObservations =
      candidate.importedExecution.spatial()->functionalObservations;
  if (!loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          dfgObservations, candidateObservations) ||
      !loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          invocation.systemDfgBoundary.functionalObservations,
          dfgObservations) ||
      (candidate.engine == Engine::Cgra &&
       !loom::sim::haveExactlyEqualSpatialFunctionalObservations(
           invocation.systemCgraBoundary.functionalObservations,
           candidateObservations)))
    return invalid("standalone Spatial engine observations differ");
  return llvm::Error::success();
}

llvm::Error writeManifest(
    llvm::StringRef workspace,
    const loom::application::FinalizedApplicationRuntimeManifest
        &applicationManifest,
    const loom::deployment::FinalizedDeployment &deployment,
    const loom::runtime::FinalizedGem5SimulationBinding &binding,
    const PublishedInputs &inputs, const CompletedRun &dfg,
    const CompletedRun &cgra,
    llvm::ArrayRef<SpatialInvocationCase> spatialInvocations,
    llvm::ArrayRef<CompletedSpatialRun> spatialRuns,
    const ResourceTimeDriveOutcome &drive,
    const loom::application::FinalizedApplicationResourceTimeExecutionTrace
        *resourceTimeTrace,
    const loom::deployment::FinalizedDeployment *mappedRtlDeployment) {
  std::string body;
  llvm::raw_string_ostream stream(body);
  llvm::json::OStream json(stream, 2);
  json.object([&] {
    json.attribute("schema", "loom.execution_matrix_workspace.2.0");
    json.attributeObject("application_runtime_manifest", [&] {
      loom::writeArtifactRootReferenceJsonFields(
          json, applicationManifest.reference());
    });
    json.attributeObject("deployment", [&] {
      loom::writeArtifactRootReferenceJsonFields(json, deployment.reference());
    });
    if (mappedRtlDeployment)
      json.attributeObject("mapped_rtl_deployment", [&] {
        loom::writeArtifactRootReferenceJsonFields(
            json, mappedRtlDeployment->reference());
      });
    json.attributeObject("gem5_binding", [&] {
      loom::writeArtifactRootReferenceJsonFields(json, binding.reference());
    });
    json.attributeObject("workload", [&] {
      loom::writeArtifactRootReferenceJsonFields(json, inputs.workload);
    });
    json.attributeObject("runtime_input", [&] {
      loom::writeArtifactRootReferenceJsonFields(json, inputs.runtimeInput);
    });
    json.attributeBegin("product_profile");
    if (applicationManifest.manifest().productOracle()) {
      const auto &product =
          *applicationManifest.manifest().productOracle();
      json.object([&] {
        json.attribute("entry_abi",
                       loom::application::productEntryAbiSpelling(
                           product.entryAbi));
        json.attribute("entry_symbol", product.entrySymbol);
        json.attribute("warmup_samples", product.warmupSamples);
        json.attribute("measured_samples", product.measuredSamples);
        json.attribute("measured_output_bytes_per_sample",
                       product.measuredOutputBytesPerSample);
        json.attribute("expected_output_sha256",
                       loom::formatBlobDigestHex(product.expectedOutput));
        json.attribute("output_interface_ordinal",
                       product.outputInterfaceOrdinal);
      });
    } else {
      json.value(nullptr);
    }
    json.attributeEnd();
    if (const auto *synchronous = std::get_if<ResourceTimeDrive>(&drive))
      json.attributeObject("resource_time_drive", [&] {
        json.attribute("status", "synchronous");
        json.attribute("engine", engineSpelling(Engine::Dfg));
        json.attribute("endpoint_count",
                       synchronous->endpoints.deployments.size());
        json.attribute("acknowledged_event_count",
                       synchronous->acknowledgements.size());
      });
    else if (const auto *refusal =
                 std::get_if<ResourceTimeDriveRefusal>(&drive))
      json.attributeObject("resource_time_drive", [&] {
        json.attribute("status", "unsupported");
        json.attribute("reason",
                       loom::runtime::gem5RootEventControlErrorReasonSpelling(
                           refusal->reason));
      });
    if (resourceTimeTrace) {
      json.attributeObject("resource_time_execution_trace", [&] {
        loom::writeArtifactRootReferenceJsonFields(
            json, resourceTimeTrace->reference());
      });
      json.attribute("resource_time_event_count",
                     resourceTimeTrace->events().size());
    }
    json.attributeArray("runs", [&] {
      for (const CompletedRun *run : {&dfg, &cgra}) {
        json.object([&] {
          json.attribute("scope", "system");
          json.attribute("engine", engineSpelling(run->engine));
          json.attributeObject("request", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run->request);
          });
          json.attributeObject("evidence", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run->evidence);
          });
          json.attributeObject("execution", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run->execution);
          });
          json.attributeBegin("product_oracle_request");
          if (run->productOracleRequest)
            loom::writeArtifactRootReferenceJson(json,
                                                 *run->productOracleRequest);
          else
            json.value(nullptr);
          json.attributeEnd();
          json.attributeBegin("product_oracle_evidence");
          if (run->productOracleEvidence)
            loom::writeArtifactRootReferenceJson(json,
                                                 *run->productOracleEvidence);
          else
            json.value(nullptr);
          json.attributeEnd();
          const auto &progress =
              run->importedExecution.system()->progressObservations;
          json.attribute("entry_tick", progress.programEntryAccepted.gem5Tick);
          json.attribute("exit_tick",
                         progress.programExitVisible
                             ? progress.programExitVisible->gem5Tick
                             : 0);
          json.attribute("terminal_tick", progress.terminalObserved.gem5Tick);
        });
      }
      for (const CompletedSpatialRun &run : spatialRuns) {
        const SpatialInvocationCase &invocation =
            spatialInvocations[run.invocationOrdinal];
        json.object([&] {
          json.attribute("scope", "spatial");
          json.attribute("engine", engineSpelling(run.engine));
          json.attribute("invocation_ordinal", run.invocationOrdinal);
          json.attribute("dispatch_target_ordinal",
                         invocation.dispatchTargetOrdinal);
          json.attribute("acc_core_ref", invocation.accCoreReference);
          json.attribute("execution_context_key",
                         invocation.executionContextKey);
          json.attributeArray("dense_coordinates", [&] {
            for (std::uint64_t coordinate : invocation.denseCoordinates)
              json.value(coordinate);
          });
          json.attributeObject("dataflow", [&] {
            loom::writeArtifactRootReferenceJsonFields(json,
                                                       invocation.dataflow);
          });
          json.attributeObject("spatial_mapping", [&] {
            loom::writeArtifactRootReferenceJsonFields(
                json, invocation.spatialMapping);
          });
          json.attributeObject("hardware_implementation", [&] {
            loom::writeArtifactRootReferenceJsonFields(
                json, invocation.hardwareImplementation);
          });
          json.attributeObject("workload", [&] {
            loom::writeArtifactRootReferenceJsonFields(json,
                                                       invocation.workload);
          });
          json.attributeObject("runtime_input", [&] {
            loom::writeArtifactRootReferenceJsonFields(json,
                                                       invocation.runtimeInput);
          });
          json.attributeObject("request", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run.request);
          });
          json.attributeObject("evidence", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run.evidence);
          });
          json.attributeObject("execution", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run.execution);
          });
          const auto &progress =
              run.importedExecution.spatial()->progressObservations;
          json.attributeObject("terminal_cycle", [&] {
            json.attribute(
                "numerator",
                progress.terminalObserved.referenceCycle.numerator());
            json.attribute(
                "denominator",
                progress.terminalObserved.referenceCycle.denominator());
            json.attribute("delta", progress.terminalObserved.delta);
          });
        });
      }
    });
    const auto &results =
        dfg.importedExecution.system()->functionalObservations.valueResults;
    json.attributeArray("value_results", [&] {
      for (const loom::sim::ValueResultObservation &result : results) {
        const auto *published =
            std::get_if<loom::sim::PublishedValueResult>(&result);
        if (!published) {
          json.value(nullptr);
          continue;
        }
        json.array([&] {
          for (const loom::sim::SemanticLane &lane : published->value.lanes) {
            if (lane.state != loom::sim::SemanticState::Defined) {
              json.value(lane.state == loom::sim::SemanticState::Poison
                             ? "poison"
                             : "undef");
              continue;
            }
            llvm::SmallString<64> digits;
            lane.bits.toStringUnsigned(digits, 16);
            json.value(digits);
          }
        });
      }
    });
  });
  std::error_code error;
  llvm::raw_fd_ostream output(child(workspace, "manifest.json"), error,
                              llvm::sys::fs::OF_Text);
  if (error)
    return ioError("cannot create execution manifest");
  output << body << '\n';
  output.close();
  if (output.has_error())
    return ioError("cannot write execution manifest");
  return llvm::Error::success();
}

llvm::Error run() {
  if (!mappedRtl && (mappedRtlLocalToolConfigPath.getNumOccurrences() != 0 ||
                     mappedRtlProviderBuild.getNumOccurrences() != 0))
    return invalid("mapped-RTL tool configuration requires --mapped-rtl");
  std::optional<loom::system_run::MappedRtlProviderOptions>
      mappedRtlProviderOptions;
  if (mappedRtl) {
    auto loaded = loadMappedRtlProviderOptions();
    if (!loaded)
      return loaded.takeError();
    mappedRtlProviderOptions.emplace(std::move(*loaded));
  }
  const bool profileRequested = !spatialCgraProfileOutput.empty() ||
                                spatialCgraWarmupRuns != 0 ||
                                spatialCgraMeasurementRuns != 0;
  if (profileRequested &&
      (spatialCgraProfileOutput.empty() || spatialCgraWarmupRuns == 0 ||
       spatialCgraMeasurementRuns == 0))
    return invalid("Spatial CGRA profiling requires an output, warmups, and "
                   "measurements");
  constexpr std::uint64_t maximumProfileRuns = 16;
  if (spatialCgraWarmupRuns > maximumProfileRuns ||
      spatialCgraMeasurementRuns > maximumProfileRuns)
    return invalid("Spatial CGRA profiling run count exceeds its bounded "
                   "interface");
  if (llvm::Error error = loom::dse::registerProductionDseOwners())
    return error;
  if (mappedRtl)
    if (llvm::Error error =
            loom::eda::open_source::registerMappedRtlSimulationProvider())
      return error;
  if (llvm::Error error = loom::runtime::registerRuntimeProvider(
          loom::runtime::fabricModelRuntimeProviderDescriptor()))
    return error;
  auto workspace = initializeWorkspace();
  if (!workspace)
    return workspace.takeError();
  const std::string &workspacePath = workspace->path;
  loom::ArtifactStore artifacts(child(workspacePath, "objects"));
  loom::BlobStore blobs(child(workspacePath, "blobs"));
  // One immutable gem5 facts session for the workspace's store lifetime: every
  // System cell's prepare and import reuse its verified closures and proofs.
  loom::runtime::Gem5SystemFactsSession gem5FactsSession(artifacts, blobs);
  loom::evaluation::ArtifactImportCacheScope artifactImportSession(
      artifacts, &blobs, invocationArtifactImportEntries);
  loom::fabric::FabricArtifactImportSession fabricImportSession;
  loom::hardware::ConfigurationABIImportSession configurationAbiImportSession;
  loom::mapping::SystemMappingImportSession systemMappingImportSession(
      artifacts, invocationSystemMappingEntries);
  loom::deployment::ConfigurationImageProjectionSession projectionSession(
      artifacts, invocationConfigurationProjectionEntries);

  llvm::Error result = [&]() -> llvm::Error {
    const loom::deployment::FinalizedDeployment &deployment =
        workspace->package.deployment();
    const loom::application::ApplicationRuntimeManifest &manifest =
        workspace->package.manifest().manifest();
    if (manifest.productOracle() && expectedI32.getNumOccurrences() != 0)
      return invalid("expected-i32 cannot replace a manifest product oracle");
    std::error_code directoryError;
    if (!std::filesystem::create_directory(
            std::filesystem::path(workspacePath) / "bundles", directoryError) ||
        directoryError)
      return ioError("cannot create bundle directory");
    auto readiness = readReadiness();
    if (!readiness)
      return readiness.takeError();
    auto binding = loom::runtime::finalizeBuiltinGem5SimulationBinding(
        manifest.selectedSystem(), readiness->identity, {}, artifacts);
    if (!binding)
      return binding.takeError();
    auto inputs = loadInputs(manifest, deployment, artifacts, blobs);
    if (!inputs)
      return inputs.takeError();
    auto resolution =
        buildResolution(deployment, *binding, *inputs, artifacts, blobs);
    if (!resolution)
      return resolution.takeError();
    auto drive = prepareResourceTimeDrive(workspace->package.manifest(),
                                          deployment, artifacts, blobs);
    if (!drive)
      return drive.takeError();
    ResourceTimeDrive *synchronousDrive =
        std::get_if<ResourceTimeDrive>(&*drive);
    auto dfg = execute(Engine::Dfg, workspacePath, deployment, *binding,
                       *inputs, *resolution, *readiness, artifacts, blobs,
                       synchronousDrive);
    if (!dfg)
      return dfg.takeError();
    auto cgra = execute(Engine::Cgra, workspacePath, deployment, *binding,
                        *inputs, *resolution, *readiness, artifacts, blobs,
                        nullptr);
    if (!cgra)
      return cgra.takeError();
    if (llvm::Error error = validateSystemResults(*dfg, *cgra))
      return error;
    if (manifest.productOracle()) {
      if (llvm::Error error = publishProductOracleEvidence(
              workspace->package.manifest(), *dfg, *resolution, artifacts,
              blobs))
        return error;
      if (llvm::Error error = publishProductOracleEvidence(
              workspace->package.manifest(), *cgra, *resolution, artifacts,
              blobs))
        return error;
    }
    std::optional<
        loom::application::FinalizedApplicationResourceTimeExecutionTrace>
        resourceTimeTrace;
    if (synchronousDrive) {
      auto trace =
          publishResourceTimeDriveTrace(*synchronousDrive, artifacts, blobs);
      if (!trace)
        return trace.takeError();
      resourceTimeTrace.emplace(std::move(*trace));
    }
    if (dfg->spatialInvocations.size() != cgra->spatialInvocations.size())
      return invalid("System engines observed different launch counts");
    std::vector<SpatialInvocationCase> spatialInvocations;
    spatialInvocations.reserve(dfg->spatialInvocations.size());
    for (std::size_t ordinal = 0; ordinal != dfg->spatialInvocations.size();
         ++ordinal) {
      auto invocation = materializeSpatialInvocationCase(
          ordinal, dfg->spatialInvocations[ordinal],
          cgra->spatialInvocations[ordinal], deployment, artifacts, blobs);
      if (!invocation)
        return invocation.takeError();
      spatialInvocations.push_back(std::move(*invocation));
    }
    std::optional<loom::deployment::FinalizedDeployment> rtlDeployment;
    if (mappedRtl) {
      auto derived = loom::system_run::deriveMappedRtlDeployment(
          deployment, artifacts, blobs);
      if (!derived)
        return derived.takeError();
      rtlDeployment.emplace(std::move(*derived));
    }
    std::vector<CompletedSpatialRun> spatialRuns;
    spatialRuns.reserve(spatialInvocations.size() * (mappedRtl ? 3 : 2));
    std::vector<SpatialCgraProfileRecord> spatialProfiles;
    if (profileRequested)
      spatialProfiles.reserve(spatialInvocations.size());
    for (const SpatialInvocationCase &invocation : spatialInvocations) {
      auto spatialDfg =
          executeSpatial(Engine::Dfg, invocation, artifacts, blobs);
      if (!spatialDfg)
        return spatialDfg.takeError();
      std::optional<CompletedSpatialRun> spatialCgra;
      if (profileRequested) {
        auto prepared = prepareSpatialCgra(invocation, artifacts, blobs);
        if (!prepared)
          return prepared.takeError();
        SpatialCgraProfileRecord profile;
        profile.invocationOrdinal = invocation.ordinal;
        profile.warmups.reserve(spatialCgraWarmupRuns);
        for (std::uint64_t run = 0; run != spatialCgraWarmupRuns; ++run) {
          auto profiled = executeProfiledSpatialCgra(*prepared, invocation, run,
                                                     artifacts, blobs);
          if (!profiled)
            return profiled.takeError();
          if (llvm::Error error = validateSpatialResults(
                  invocation, *spatialDfg, profiled->completed))
            return error;
          profile.warmups.push_back(std::move(profiled->sample));
        }
        profile.measurements.reserve(spatialCgraMeasurementRuns);
        for (std::uint64_t run = 0; run != spatialCgraMeasurementRuns; ++run) {
          auto profiled = executeProfiledSpatialCgra(
              *prepared, invocation, spatialCgraWarmupRuns + run, artifacts,
              blobs);
          if (!profiled)
            return profiled.takeError();
          if (llvm::Error error = validateSpatialResults(
                  invocation, *spatialDfg, profiled->completed))
            return error;
          profile.measurements.push_back(std::move(profiled->sample));
          spatialCgra.emplace(std::move(profiled->completed));
        }
        spatialProfiles.push_back(std::move(profile));
      } else {
        auto executed =
            executeSpatial(Engine::Cgra, invocation, artifacts, blobs);
        if (!executed)
          return executed.takeError();
        spatialCgra.emplace(std::move(*executed));
      }
      if (llvm::Error error =
              validateSpatialResults(invocation, *spatialDfg, *spatialCgra))
        return error;
      std::optional<CompletedSpatialRun> spatialRtl;
      if (mappedRtl) {
        auto cell = loom::system_run::executeMappedRtlCell(
            invocation, *rtlDeployment, child(workspacePath, "bundles"),
            *mappedRtlProviderOptions, artifacts, blobs);
        if (!cell)
          return cell.takeError();
        auto executed = completeSpatialRun(
            Engine::Rtl, invocation.ordinal, cell->request, cell->resolution,
            std::move(cell->evidence), artifacts, blobs);
        if (!executed)
          return executed.takeError();
        if (llvm::Error error =
                validateSpatialResults(invocation, *spatialDfg, *executed))
          return error;
        spatialRtl.emplace(std::move(*executed));
      }
      spatialRuns.push_back(std::move(*spatialDfg));
      spatialRuns.push_back(std::move(*spatialCgra));
      if (spatialRtl)
        spatialRuns.push_back(std::move(*spatialRtl));
    }
    if (llvm::Error error = writeManifest(
            workspacePath, workspace->package.manifest(), deployment, *binding,
            *inputs, *dfg, *cgra,
            spatialInvocations, spatialRuns, *drive,
            resourceTimeTrace ? &*resourceTimeTrace : nullptr,
            rtlDeployment ? &*rtlDeployment : nullptr))
      return error;
    if (profileRequested) {
      auto peakResidentBytes = processPeakResidentBytes();
      if (!peakResidentBytes)
        return peakResidentBytes.takeError();
      if (llvm::Error error = writeSpatialCgraProfile(
              spatialCgraProfileOutput, workspace->package.manifest(),
              deployment, spatialInvocations, spatialProfiles,
              *peakResidentBytes))
        return error;
    }
    return llvm::Error::success();
  }();

  loom::evaluation::emitArtifactImportCacheStatistics(
      loom::evaluation::ArtifactImportCacheVerificationDomain::SourceInvocation,
      artifactImportSession.statistics());
  loom::fabric::emitFabricArtifactImportSessionStatistics(
      loom::fabric::FabricArtifactImportVerificationDomain::SourceInvocation,
      loom::InvocationDiagnosticStage::Deployment,
      fabricImportSession.statistics());
  loom::hardware::emitConfigurationABIImportSessionStatistics(
      loom::hardware::ConfigurationABIImportVerificationDomain::
          SourceInvocation,
      configurationAbiImportSession.statistics());
  loom::deployment::emitConfigurationImageProjectionSessionStatistics(
      loom::deployment::ConfigurationImageProjectionVerificationDomain::
          SourceInvocation,
      projectionSession.statistics());
  loom::mapping::emitSystemMappingImportSessionStatistics(
      loom::mapping::SystemMappingImportVerificationDomain::SourceInvocation,
      systemMappingImportSession.statistics());
  loom::runtime::emitGem5SystemFactsSessionStatistics(
      gem5FactsSession.statistics());
  return result;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Execute a Deployment through gem5\n");
  if (llvm::Error error = run()) {
    llvm::errs() << "loom-system-run: " << llvm::toString(std::move(error))
                 << '\n';
    return 1;
  }
  llvm::outs() << child(outputWorkspace, "manifest.json") << '\n';
  return 0;
}
