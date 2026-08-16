#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Deployment/Deployment.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Deployment/Package.h"
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
#include "Hardware/Configuration/ConfigurationDiagnostics.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Runtime/FabricModelPlatform.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Runtime/SpatialInvocationWire.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialInvocation.h"
#include "Simulator/SpatialObservationComparison.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
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
                      llvm::cl::desc("Input Deployment package"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    outputWorkspace("output", llvm::cl::desc("New execution workspace"),
                    llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    gem5Readiness("gem5-readiness",
                  llvm::cl::desc("Pinned gem5 readiness JSON"),
                  llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::uint64_t>
    programEntry("program-entry",
                 llvm::cl::desc("Deployment program entry ordinal"),
                 llvm::cl::init(0));
llvm::cl::opt<std::int64_t>
    expectedI32("expected-i32",
                llvm::cl::desc("Independent expected i32 result"));

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_run_invalid: " + message);
}

llvm::Error ioError(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                 "system_run_io_error: " + message);
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

llvm::Expected<std::string> initializeWorkspace() {
  auto source = canonicalPath(deploymentPackage, true);
  if (!source)
    return source.takeError();
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
  if (llvm::Error copyError = copyRegularDirectory(
          child(*source, "objects"), (output / "objects").string()))
    return std::move(copyError);
  if (llvm::Error copyError = copyRegularDirectory(child(*source, "blobs"),
                                                   (output / "blobs").string()))
    return std::move(copyError);
  if (!std::filesystem::create_directory(output / "bundles", error) || error)
    return ioError("cannot create bundle directory");
  return output.string();
}

llvm::Expected<std::shared_ptr<const loom::deployment::FinalizedDeployment>>
importPackagedDeployment(llvm::StringRef workspace,
                         const loom::ArtifactStore &artifacts,
                         const loom::BlobStore &blobs) {
  auto rootText = readText(child(workspace, "root"));
  if (!rootText)
    return rootText.takeError();
  if (rootText->size() != 64)
    return invalid("Deployment package root is not one SHA-256 identity");
  auto identity = loom::parseArtifactIdentityHex(*rootText);
  if (!identity)
    return identity.takeError();
  loom::ArtifactRootReference reference{
      loom::deployment::deploymentSchema.identity.str(),
      loom::deployment::deploymentSchema.version, *identity};
  const std::array<loom::ArtifactRootReference, 1> references{reference};
  auto deployment = loom::evaluation::importCachedArtifact<
      loom::deployment::FinalizedDeployment>(
      artifacts, &blobs, references, [&] {
        return loom::deployment::importDeployment(reference, artifacts, blobs);
      });
  if (!deployment)
    return deployment.takeError();
  auto closure = loom::deployment::deriveDeploymentPackageClosure(
      **deployment, artifacts, blobs);
  if (!closure)
    return closure.takeError();

  std::set<std::string> expectedArtifacts;
  for (const loom::ArtifactRootReference &root : closure->artifacts())
    expectedArtifacts.insert(loom::formatArtifactIdentityHex(root.artifact));
  std::set<std::string> expectedBlobs;
  for (const loom::BlobDigest &blob : closure->blobs())
    expectedBlobs.insert(loom::formatBlobDigestHex(blob));
  auto requireEntries =
      [](llvm::StringRef directory,
         const std::set<std::string> &expected) -> llvm::Error {
    std::error_code error;
    std::set<std::string> actual;
    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(directory.str(), error)) {
      if (error || !entry.is_regular_file(error) || entry.is_symlink(error))
        return invalid("execution store contains a non-regular package entry");
      actual.insert(entry.path().filename().string());
    }
    if (error)
      return ioError("cannot enumerate execution store");
    if (actual != expected)
      return invalid("Deployment package has missing or unreferenced entries");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          requireEntries(child(workspace, "objects"), expectedArtifacts))
    return std::move(error);
  if (llvm::Error error =
          requireEntries(child(workspace, "blobs"), expectedBlobs))
    return std::move(error);
  return std::move(*deployment);
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
publishInputs(const loom::deployment::FinalizedDeployment &deployment,
              const loom::ArtifactStore &artifacts) {
  const loom::deployment::DeploymentProgramEntryRef entry{
      deployment.reference().artifact, programEntry};
  auto shapes = loom::sim::projectSystemSimulationBoundaryShapes(
      deployment, entry, artifacts);
  if (!shapes)
    return shapes.takeError();
  if (!shapes->valueArguments.empty())
    return invalid("program entry value arguments require explicit inputs");
  loom::sim::SystemSimulationWorkload draft{entry};
  draft.observableContract.valueResults.resize(shapes->valueResults.size());
  std::iota(draft.observableContract.valueResults.begin(),
            draft.observableContract.valueResults.end(), 0);
  auto workload =
      loom::sim::finalizeSimulationWorkload(draft, deployment, artifacts);
  if (!workload)
    return workload.takeError();
  loom::sim::SystemSimulationRuntimeInputDraft runtimeDraft{
      workload->identity()};
  auto runtime = loom::sim::finalizeSimulationRuntimeInput(
      runtimeDraft, *workload, deployment, artifacts);
  if (!runtime)
    return runtime.takeError();
  auto workloadRef = loom::sim::publishSimulationWorkload(*workload, artifacts);
  if (!workloadRef)
    return workloadRef.takeError();
  auto runtimeRef =
      loom::sim::publishSimulationRuntimeInput(*runtime, artifacts);
  if (!runtimeRef)
    return runtimeRef.takeError();
  return PublishedInputs{std::move(*workloadRef), std::move(*runtimeRef)};
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

enum class Engine : std::uint8_t { Dfg, Cgra };

struct ObservedSpatialInvocation final {
  std::vector<std::uint8_t> invocation;
  std::vector<std::uint8_t> boundaryResult;
};

struct CompletedRun final {
  Engine engine;
  loom::ArtifactRootReference request;
  loom::ArtifactRootReference evidence;
  loom::ArtifactRootReference execution;
  loom::sim::CanonicalSimulationExecution importedExecution;
  std::vector<ObservedSpatialInvocation> spatialInvocations;
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
  if (!bridges || bridges->empty())
    return invalid("gem5 projection contains no Spatial bridge");

  std::vector<ObservedSpatialInvocation> invocations;
  invocations.reserve(bridges->size());
  for (const auto indexed : llvm::enumerate(*bridges)) {
    const llvm::json::Object *bridge = indexed.value().getAsObject();
    if (!bridge)
      return invalid("gem5 projection contains a non-object bridge");
    const auto resultPath = bridge->getString("result_path");
    const std::string expectedPath =
        "outputs/spatial-bridge-" + std::to_string(indexed.index()) + ".result";
    if (!resultPath || *resultPath != expectedPath)
      return invalid("gem5 projection has a noncanonical bridge result path");
    auto resultText = readText(child(prepared.bundleRoot, expectedPath));
    if (!resultText)
      return resultText.takeError();
    const std::vector<std::uint8_t> resultBytes(resultText->begin(),
                                                resultText->end());
    loom::runtime::Gem5BridgeResult bridgeResult;
    std::string diagnostic;
    if (!loom::runtime::decodeGem5BridgeResult(resultBytes, bridgeResult,
                                               diagnostic))
      return invalid("cannot decode verified gem5 bridge result: " +
                     llvm::Twine(diagnostic));
    if (bridgeResult.status != 0 || bridgeResult.sequence != 0)
      return invalid("gem5 bridge did not retire its first invocation");
    loom::runtime::SpatialInvocationResultWire invocationResult;
    if (!loom::runtime::decodeSpatialInvocationResultWire(
            bridgeResult.result, invocationResult, diagnostic))
      return invalid("cannot decode verified Spatial invocation result: " +
                     llvm::Twine(diagnostic));
    if (invocationResult.invocation.empty())
      return invalid("public execution matrix requires a dynamic invocation");
    invocations.push_back({std::move(invocationResult.invocation),
                           std::move(invocationResult.spatialBoundaryResult)});
  }
  return invocations;
}

llvm::Expected<CompletedRun>
execute(Engine engine, llvm::StringRef workspace,
        const loom::deployment::FinalizedDeployment &deployment,
        const loom::runtime::FinalizedGem5SimulationBinding &binding,
        const PublishedInputs &inputs,
        const loom::evaluation::CaseArtifactResolution &resolution,
        const Readiness &readiness, const loom::ArtifactStore &artifacts,
        const loom::BlobStore &blobs) {
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
  auto preparation = loom::evaluation::prepareEvaluationModelInvocation(
      *request, resolution, artifacts, blobs,
      {std::move(local), child(child(workspace, "bundles"), engineName)});
  if (!preparation)
    return preparation.takeError();
  const auto *prepared =
      std::get_if<loom::external_tool::PreparedExternalToolInvocation>(
          &*preparation);
  if (!prepared)
    return invalid(engineName + " is unsupported for the exact Deployment");
  auto status =
      loom::external_tool::executeExternalToolInvocationBundle(*prepared);
  if (!status)
    return status.takeError();
  if (*status != 0)
    return invalid(engineName + " external invocation exited with status " +
                   llvm::Twine(*status));
  auto evidence = loom::evaluation::importEvaluationModelInvocation(
      *request, resolution, *prepared, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  if (evidence->outcomeKind() !=
      loom::evaluation::EvidenceOutcomeKind::Completed)
    return invalid(engineName + " did not publish completed Evidence");
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
  auto spatialInvocations = readSpatialInvocations(*prepared);
  if (!spatialInvocations)
    return spatialInvocations.takeError();
  return CompletedRun{
      engine,       std::move(*requestRef), std::move(*evidenceRef),
      executionRef, std::move(*imported),   std::move(*spatialInvocations)};
}

struct SpatialInvocationCase final {
  std::size_t ordinal = 0;
  loom::ArtifactRootReference dataflow;
  loom::ArtifactRootReference workload;
  loom::ArtifactRootReference runtimeInput;
  loom::ArtifactRootReference hardwareImplementation;
  loom::ArtifactRootReference fabric;
  loom::ArtifactRootReference spatialMapping;
  loom::sim::SpatialEngineBoundaryResult systemDfgBoundary;
  loom::sim::SpatialEngineBoundaryResult systemCgraBoundary;
};

struct CompletedSpatialRun final {
  Engine engine;
  std::size_t invocationOrdinal = 0;
  loom::ArtifactRootReference request;
  loom::ArtifactRootReference evidence;
  loom::ArtifactRootReference execution;
  loom::sim::CanonicalSimulationExecution importedExecution;
};

bool sameWrites(llvm::ArrayRef<loom::sim::SpatialInvocationMemoryWrite> lhs,
                llvm::ArrayRef<loom::sim::SpatialInvocationMemoryWrite> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs, [](const auto &left, const auto &right) {
           return left.address == right.address && left.bytes == right.bytes;
         });
}

llvm::Expected<SpatialInvocationCase> materializeSpatialInvocationCase(
    std::size_t ordinal, const ObservedSpatialInvocation &dfg,
    const ObservedSpatialInvocation &cgra,
    const loom::deployment::FinalizedDeployment &deployment,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  if (dfg.invocation != cgra.invocation)
    return invalid("System DFG and CGRA observed different invocations");
  loom::runtime::SpatialInvocationWire wire;
  std::string diagnostic;
  if (!loom::runtime::decodeSpatialInvocationWire(dfg.invocation, wire,
                                                  diagnostic))
    return invalid("cannot decode System Spatial invocation: " +
                   llvm::Twine(diagnostic));
  auto dataflowIdentity =
      loom::ArtifactIdentity::fromBytes(wire.canonicalDataflowIdentity);
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  loom::ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  auto dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  const dataflow::RootedGraphLaunchRef graph{
      {dataflowReference.artifact,
       dataflow::RootThreadLaunchId(wire.rootThreadLaunchEntity)},
      {dataflowReference.artifact,
       dataflow::StaticGraphLaunchId(wire.graphLaunchEntity)}};
  auto shapes =
      loom::sim::projectSpatialSimulationBoundaryShapes(*dataflowView, graph);
  if (!shapes)
    return shapes.takeError();

  loom::sim::SpatialSimulationWorkload workloadDraft{graph};
  workloadDraft.denseCoordinates = wire.denseCoordinates;
  workloadDraft.valueInputPlan.assign(shapes->valueInputs.size(),
                                      loom::sim::RuntimeValueInput{});
  workloadDraft.observableContract.valueResults.resize(
      shapes->valueResults.size());
  std::iota(workloadDraft.observableContract.valueResults.begin(),
            workloadDraft.observableContract.valueResults.end(), 0);
  auto workload =
      loom::sim::finalizeSimulationWorkload(workloadDraft, *dataflowView);
  if (!workload)
    return workload.takeError();
  auto workloadReference =
      loom::sim::publishSimulationWorkload(*workload, artifacts);
  if (!workloadReference)
    return workloadReference.takeError();
  auto inputs = loom::sim::materializeSpatialInvocationInputs(
      {std::move(*dataflow), std::move(*workload)}, wire);
  if (!inputs)
    return inputs.takeError();
  auto runtimeReference =
      loom::sim::publishSimulationRuntimeInput(inputs->runtimeInput, artifacts);
  if (!runtimeReference)
    return runtimeReference.takeError();
  auto selection = loom::deployment::resolveDeploymentSpatialLaunchSelection(
      deployment, graph, wire.denseCoordinates, artifacts, blobs);
  if (!selection)
    return selection.takeError();
  auto cgraCase = loom::evaluation::models::resolveCgraSimulationCase(
      selection->spatialMapping, *workloadReference, *runtimeReference,
      artifacts);
  if (!cgraCase)
    return cgraCase.takeError();
  if (cgraCase->canonicalDataflow != dataflowReference)
    return invalid("Deployment selected a foreign Dataflow owner");

  auto dfgBoundary =
      loom::sim::decodeSpatialEngineBoundaryResult(dfg.boundaryResult, *inputs);
  if (!dfgBoundary)
    return dfgBoundary.takeError();
  auto cgraBoundary = loom::sim::decodeSpatialEngineBoundaryResult(
      cgra.boundaryResult, *inputs);
  if (!cgraBoundary)
    return cgraBoundary.takeError();
  if (!std::holds_alternative<loom::sim::RetiredExecution>(
          dfgBoundary->terminal) ||
      !std::holds_alternative<loom::sim::RetiredExecution>(
          cgraBoundary->terminal) ||
      !loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          dfgBoundary->functionalObservations,
          cgraBoundary->functionalObservations))
    return invalid("System Spatial DFG and CGRA observations differ");
  auto dfgWrites = loom::sim::projectSpatialInvocationResultWrites(
      wire, *inputs, dfgBoundary->functionalObservations);
  if (!dfgWrites)
    return dfgWrites.takeError();
  auto cgraWrites = loom::sim::projectSpatialInvocationResultWrites(
      wire, *inputs, cgraBoundary->functionalObservations);
  if (!cgraWrites)
    return cgraWrites.takeError();
  if (!sameWrites(*dfgWrites, *cgraWrites))
    return invalid("System Spatial engines projected different guest writes");

  return SpatialInvocationCase{ordinal,
                               std::move(dataflowReference),
                               std::move(*workloadReference),
                               std::move(*runtimeReference),
                               std::move(selection->hardwareImplementation),
                               std::move(cgraCase->fabric),
                               std::move(selection->spatialMapping),
                               std::move(*dfgBoundary),
                               std::move(*cgraBoundary)};
}

llvm::Expected<CompletedSpatialRun>
completeSpatialRun(Engine engine, std::size_t invocationOrdinal,
                   const loom::evaluation::EvaluationRequest &request,
                   const loom::evaluation::CaseArtifactResolution &resolution,
                   loom::evaluation::EvaluationEvidence evidence,
                   const loom::ArtifactStore &artifacts,
                   const loom::BlobStore &blobs) {
  const llvm::StringRef engineName =
      engine == Engine::Dfg ? "spatial-dfg" : "spatial-cgra";
  if (evidence.outcomeKind() !=
      loom::evaluation::EvidenceOutcomeKind::Completed)
    return invalid(engineName + " did not publish completed Evidence");
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
  auto prepared = loom::evaluation::models::prepareCgraSimulationEvaluation(
      invocation.dataflow, invocation.fabric, invocation.spatialMapping,
      invocation.workload, invocation.runtimeInput,
      loom::defaultResolvedConfig(), artifacts, blobs);
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

llvm::Error validateSystemResults(const CompletedRun &dfg,
                                  const CompletedRun &cgra) {
  const auto &lhs = dfg.importedExecution.system()->functionalObservations;
  const auto &rhs = cgra.importedExecution.system()->functionalObservations;
  if (!loom::sim::haveExactlyEqualSystemFunctionalObservations(lhs, rhs) ||
      !lhs.externalValueOutputs.empty() || !rhs.externalValueOutputs.empty() ||
      !lhs.externalStreamOutputs.empty() ||
      !rhs.externalStreamOutputs.empty() || !lhs.memories.empty() ||
      !rhs.memories.empty())
    return invalid("System DFG and CGRA functional observations differ");
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

llvm::Error validateSpatialResults(const SpatialInvocationCase &invocation,
                                   const CompletedSpatialRun &dfg,
                                   const CompletedSpatialRun &cgra) {
  if (dfg.invocationOrdinal != invocation.ordinal ||
      cgra.invocationOrdinal != invocation.ordinal ||
      dfg.engine != Engine::Dfg || cgra.engine != Engine::Cgra)
    return invalid("Spatial execution is attached to the wrong matrix cell");
  const auto &dfgObservations =
      dfg.importedExecution.spatial()->functionalObservations;
  const auto &cgraObservations =
      cgra.importedExecution.spatial()->functionalObservations;
  if (!loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          dfgObservations, cgraObservations) ||
      !loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          invocation.systemDfgBoundary.functionalObservations,
          dfgObservations) ||
      !loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          invocation.systemCgraBoundary.functionalObservations,
          cgraObservations))
    return invalid("standalone and System Spatial observations differ");
  return llvm::Error::success();
}

llvm::Error
writeManifest(llvm::StringRef workspace,
              const loom::deployment::FinalizedDeployment &deployment,
              const loom::runtime::FinalizedGem5SimulationBinding &binding,
              const PublishedInputs &inputs, const CompletedRun &dfg,
              const CompletedRun &cgra,
              llvm::ArrayRef<SpatialInvocationCase> spatialInvocations,
              llvm::ArrayRef<CompletedSpatialRun> spatialRuns) {
  std::string body;
  llvm::raw_string_ostream stream(body);
  llvm::json::OStream json(stream, 2);
  json.object([&] {
    json.attribute("schema", "loom.execution_matrix_workspace.1.0");
    json.attributeObject("deployment", [&] {
      loom::writeArtifactRootReferenceJsonFields(json, deployment.reference());
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
    json.attributeArray("runs", [&] {
      for (const CompletedRun *run : {&dfg, &cgra}) {
        json.object([&] {
          json.attribute("scope", "system");
          json.attribute("engine", run->engine == Engine::Dfg ? "dfg" : "cgra");
          json.attributeObject("request", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run->request);
          });
          json.attributeObject("evidence", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run->evidence);
          });
          json.attributeObject("execution", [&] {
            loom::writeArtifactRootReferenceJsonFields(json, run->execution);
          });
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
          json.attribute("engine", run.engine == Engine::Dfg ? "dfg" : "cgra");
          json.attribute("invocation_ordinal", run.invocationOrdinal);
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
  if (llvm::Error error =
          loom::evaluation::registerProductionEvaluationRegistry())
    return error;
  if (llvm::Error error = loom::runtime::registerRuntimeProvider(
          loom::runtime::fabricModelRuntimeProviderDescriptor()))
    return error;
  auto workspace = initializeWorkspace();
  if (!workspace)
    return workspace.takeError();
  loom::ArtifactStore artifacts(child(*workspace, "objects"));
  loom::BlobStore blobs(child(*workspace, "blobs"));
  loom::evaluation::ArtifactImportCacheScope artifactImportSession(
      artifacts, &blobs, invocationArtifactImportEntries);
  loom::fabric::FabricArtifactImportSession fabricImportSession;
  loom::hardware::ConfigurationABIImportSession configurationAbiImportSession;
  loom::mapping::SystemMappingImportSession systemMappingImportSession(
      artifacts, invocationSystemMappingEntries);
  loom::deployment::ConfigurationImageProjectionSession projectionSession(
      artifacts, invocationConfigurationProjectionEntries);

  llvm::Error result = [&]() -> llvm::Error {
    auto deployment = importPackagedDeployment(*workspace, artifacts, blobs);
    if (!deployment)
      return deployment.takeError();
    auto readiness = readReadiness();
    if (!readiness)
      return readiness.takeError();
    auto systemMapping = loom::mapping::importSystemMapping(
        (*deployment)->deployment().systemMapping(), artifacts);
    if (!systemMapping)
      return systemMapping.takeError();
    const loom::ArtifactRootReference system{
        loom::fabric::fabricArtifactSchema.identity.str(),
        loom::fabric::fabricArtifactSchema.version,
        systemMapping->view().fabricIdentity()};
    auto binding = loom::runtime::finalizeBuiltinGem5SimulationBinding(
        system, readiness->identity, {}, artifacts);
    if (!binding)
      return binding.takeError();
    auto inputs = publishInputs(**deployment, artifacts);
    if (!inputs)
      return inputs.takeError();
    auto resolution =
        buildResolution(**deployment, *binding, *inputs, artifacts, blobs);
    if (!resolution)
      return resolution.takeError();
    auto dfg = execute(Engine::Dfg, *workspace, **deployment, *binding, *inputs,
                       *resolution, *readiness, artifacts, blobs);
    if (!dfg)
      return dfg.takeError();
    auto cgra = execute(Engine::Cgra, *workspace, **deployment, *binding,
                        *inputs, *resolution, *readiness, artifacts, blobs);
    if (!cgra)
      return cgra.takeError();
    if (llvm::Error error = validateSystemResults(*dfg, *cgra))
      return error;
    if (dfg->spatialInvocations.size() != cgra->spatialInvocations.size())
      return invalid("System engines observed different launch counts");
    std::vector<SpatialInvocationCase> spatialInvocations;
    spatialInvocations.reserve(dfg->spatialInvocations.size());
    for (std::size_t ordinal = 0; ordinal != dfg->spatialInvocations.size();
         ++ordinal) {
      auto invocation = materializeSpatialInvocationCase(
          ordinal, dfg->spatialInvocations[ordinal],
          cgra->spatialInvocations[ordinal], **deployment, artifacts, blobs);
      if (!invocation)
        return invocation.takeError();
      spatialInvocations.push_back(std::move(*invocation));
    }
    std::vector<CompletedSpatialRun> spatialRuns;
    spatialRuns.reserve(spatialInvocations.size() * 2);
    for (const SpatialInvocationCase &invocation : spatialInvocations) {
      auto spatialDfg =
          executeSpatial(Engine::Dfg, invocation, artifacts, blobs);
      if (!spatialDfg)
        return spatialDfg.takeError();
      auto spatialCgra =
          executeSpatial(Engine::Cgra, invocation, artifacts, blobs);
      if (!spatialCgra)
        return spatialCgra.takeError();
      if (llvm::Error error =
              validateSpatialResults(invocation, *spatialDfg, *spatialCgra))
        return error;
      spatialRuns.push_back(std::move(*spatialDfg));
      spatialRuns.push_back(std::move(*spatialCgra));
    }
    return writeManifest(*workspace, **deployment, *binding, *inputs, *dfg,
                         *cgra, spatialInvocations, spatialRuns);
  }();

  loom::evaluation::emitArtifactImportCacheStatistics(
      loom::evaluation::ArtifactImportCacheVerificationDomain::SourceInvocation,
      artifactImportSession.statistics());
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
