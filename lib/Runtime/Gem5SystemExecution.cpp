#include "Runtime/Gem5SystemExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Deployment/Deployment.h"
#include "Deployment/DeploymentReference.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "Deployment/Package.h"
#include "EDA/Adapters/OpenSource/MappedRtlExecution.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#ifndef LOOM_GEM5_CONFIG_SCRIPT_PATH
#error "LOOM_GEM5_CONFIG_SCRIPT_PATH is required"
#endif
#ifndef LOOM_GEM5_DFG_ENGINE_PATH
#error "LOOM_GEM5_DFG_ENGINE_PATH is required"
#endif
#ifndef LOOM_GEM5_CGRA_ENGINE_PATH
#error "LOOM_GEM5_CGRA_ENGINE_PATH is required"
#endif
#ifndef LOOM_GEM5_RTL_ENGINE_SOURCE_PATH
#error "LOOM_GEM5_RTL_ENGINE_SOURCE_PATH is required"
#endif
#ifndef LOOM_GEM5_BRIDGE_HEADER_PATH
#error "LOOM_GEM5_BRIDGE_HEADER_PATH is required"
#endif

namespace loom::runtime {
namespace {

using namespace evaluation;
using namespace external_tool;

constexpr CaseSubjectRoleRef kDeploymentRole(0);
constexpr CaseSubjectRoleRef kBindingRole(1);
constexpr ModelOutputSlotRef kExecutionOutput(0);
constexpr llvm::StringLiteral kSystemResultPath =
    "outputs/system-result.json";
constexpr llvm::StringLiteral kBridgeResultPath =
    "outputs/spatial-bridge-0.result";
constexpr llvm::StringLiteral kProjectionPath =
    "drivers/gem5-system-projection.json";
constexpr llvm::StringLiteral kConfigurationScriptPath =
    "drivers/configure_loom_system.py";
constexpr llvm::StringLiteral kDfgEnginePath =
    "drivers/loom-gem5-dfg-engine";
constexpr llvm::StringLiteral kCgraEnginePath =
    "drivers/loom-gem5-cgra-engine";
constexpr llvm::StringLiteral kBridgeHeaderPath =
    "drivers/Gem5BridgeWire.h";
constexpr llvm::StringLiteral kPackageObjectPath = "inputs/package/objects";
constexpr llvm::StringLiteral kHostElfPath = "inputs/host.elf";
constexpr llvm::StringLiteral kSpatialLaunchPath =
    "inputs/spatial-launch.bin";
constexpr llvm::StringLiteral kThreadDispatchPath =
    "inputs/thread-dispatch.bin";
constexpr llvm::StringLiteral kAdmissionPath = "inputs/admission.bin";
constexpr std::uint64_t kMaximumGem5Ticks = 10'000'000;
constexpr std::uint64_t kMaximumSpatialWork = 1'000'000;
constexpr std::uint64_t kGem5PageBytes = 4096;
constexpr std::uint64_t kGem5StackBytes = 64 * 1024;
constexpr std::uint64_t kThreadDispatchApertureBytes = 4096;

enum class Gem5SystemEngine { Dfg, Cgra, Rtl };

struct Gem5ProcessorProjection final {
  Gem5ProcessorFabricRef processor;
  Gem5RiscvTimingCpuParameters parameters;
};

struct Gem5InstructionImage final {
  ArtifactRootReference reference;
  std::string path;
};

struct Gem5RuntimeImage final {
  std::string path;
  std::uint64_t address = 0;
};

struct Gem5DispatchTarget final {
  std::uint64_t cpuId = 0;
  std::uint64_t imageOrdinal = 0;
  std::string entrySymbol;
  std::uint64_t bridgeAddress = 0;
  std::uint64_t launchAddress = 0;
  std::uint64_t launchSize = 0;
};

struct ReadinessIdentity final {
  std::string binarySha256;
  ExternalFileFingerprint binaryFingerprint;
};

struct Gem5SystemFacts final {
  Gem5SystemEngine engine;
  ArtifactRootReference deployment;
  ArtifactRootReference binding;
  ArtifactRootReference dataflow;
  ArtifactRootReference fabric;
  ArtifactRootReference spatialMapping;
  ArtifactRootReference hardwareImplementation;
  ArtifactRootReference spatialWorkload;
  ArtifactRootReference spatialRuntimeInput;
  std::vector<std::uint8_t> launchPayload;
  std::vector<MaterializedBundleFile> semanticInputs;
  std::vector<Gem5ProcessorProjection> processors;
  std::string hostEntrySymbol;
  std::uint64_t hostCpuId = 0;
  std::vector<Gem5InstructionImage> instructionImages;
  std::vector<Gem5RuntimeImage> runtimeImages;
  std::uint64_t dispatchAddress = 0;
  std::uint64_t stackBase = 0;
  std::uint64_t stackStride = 0;
  std::vector<Gem5DispatchTarget> dispatchTargets;
  Gem5SpatialBridgeParameters bridge;
  Gem5SimpleMemoryParameters memory;
};

using Gem5SystemFactsOrUnsupported =
    std::variant<Gem5SystemFacts, UnsupportedEvidence>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_system_execution_invalid: " + message);
}

std::string bytesToString(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

llvm::Expected<std::string> readFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return invalid("cannot read '" + path + "': " +
                   buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

Gem5SystemEngine selectedEngine(const EvaluationRequest &request) {
  const EvaluationModelKind kind = request.modelBinding().descriptorRef().modelKind();
  if (kind == builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemDfg))
    return Gem5SystemEngine::Dfg;
  if (kind == builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemCgra))
    return Gem5SystemEngine::Cgra;
  return Gem5SystemEngine::Rtl;
}

llvm::Expected<std::pair<ArtifactRootReference, ArtifactRootReference>>
systemSubjects(const EvaluationRequest &request) {
  const auto deployments = request.subjectBindings().subjects(kDeploymentRole);
  const auto bindings = request.subjectBindings().subjects(kBindingRole);
  if (deployments.size() != 1 || bindings.size() != 1)
    return invalid("Request does not bind one Deployment and one gem5 binding");
  return std::pair(deployments.front(), bindings.front());
}

bool isEmptySystemSurface(const sim::ImportedSystemSimulationInputs &inputs) {
  const sim::SystemSimulationWorkload &workload = *inputs.workload.system();
  const sim::SystemSimulationRuntimeInput &runtime = *inputs.runtimeInput.system();
  return workload.valueInputPlan.empty() &&
         workload.externalValueInputPlan.empty() &&
         workload.observableContract.valueResults.empty() &&
         workload.observableContract.externalValueOutputs.empty() &&
         workload.observableContract.externalStreamOutputs.empty() &&
         workload.observableContract.memories.empty() &&
         runtime.runtimeEntryValues.empty() &&
         runtime.runtimeExternalValues.empty() &&
         runtime.externalStreamInputs.empty() && runtime.memoryObjects.empty() &&
         runtime.memoryInterfaceBindings.empty();
}

llvm::Expected<std::vector<MaterializedBundleFile>>
materializeDeploymentPackage(const deployment::FinalizedDeployment &deployment,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  llvm::SmallString<256> root;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-gem5-deployment-package-%%%%%%", root))
    return invalid("cannot create Deployment package staging directory: " +
                   error.message());
  const std::filesystem::path staging(root.str().str());
  llvm::scope_exit cleanup([&] { std::filesystem::remove_all(staging); });
  const std::filesystem::path package = staging / "package";
  if (llvm::Error error = deployment::publishDeploymentPackage(
          deployment, package.string(), artifacts, blobs))
    return std::move(error);

  std::vector<std::filesystem::path> paths;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator iterator(package, error),
       end;
       !error && iterator != end; iterator.increment(error)) {
    const auto status = iterator->symlink_status(error);
    if (error)
      break;
    if (std::filesystem::is_symlink(status) ||
        (!std::filesystem::is_directory(status) &&
         !std::filesystem::is_regular_file(status)))
      return invalid("Deployment package contains a non-ordinary entry");
    if (std::filesystem::is_regular_file(status))
      paths.push_back(iterator->path());
  }
  if (error)
    return invalid("cannot enumerate Deployment package: " + error.message());
  llvm::sort(paths);

  std::vector<MaterializedBundleFile> files;
  files.reserve(paths.size());
  for (const std::filesystem::path &path : paths) {
    auto contents = readFile(path.string());
    if (!contents)
      return contents.takeError();
    files.push_back({"inputs/package/" +
                         path.lexically_relative(package).generic_string(),
                     std::move(*contents), deployment.reference(), false});
  }
  return files;
}

llvm::Error appendStoredObject(std::vector<MaterializedBundleFile> &files,
                               const ArtifactRootReference &reference,
                               const ArtifactStore &artifacts) {
  const std::string path = "inputs/package/objects/" +
                           formatArtifactIdentityHex(reference.artifact);
  if (llvm::any_of(files, [&](const MaterializedBundleFile &file) {
        return file.relativePath == path;
      }))
    return llvm::Error::success();
  auto object = artifacts.getStoredObject(reference);
  if (!object)
    return object.takeError();
  files.push_back(
      {path, bytesToString(*object), reference, false});
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> checkedAdd(std::uint64_t lhs,
                                        std::uint64_t rhs,
                                        llvm::StringRef role) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    return invalid(role + " address range overflows uint64");
  return lhs + rhs;
}

llvm::Expected<std::uint64_t> alignUp(std::uint64_t value,
                                     std::uint64_t alignment,
                                     llvm::StringRef role) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    return invalid(role + " alignment is not a power of two");
  const std::uint64_t mask = alignment - 1;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask)
    return invalid(role + " alignment overflows uint64");
  return (value + mask) & ~mask;
}

struct SelectedInstructionEntry final {
  std::size_t imageOrdinal = 0;
  std::uint64_t entryOrdinal = 0;
};

llvm::Expected<SelectedInstructionEntry> selectInstructionEntry(
    const deployment::FinalizedDeployment &deployment,
    dataflow::RootThreadLaunchRef root,
    fabric::AccCoreOccurrenceRef accCore,
    const ArtifactIdentity &fabricIdentity, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  std::optional<SelectedInstructionEntry> selected;
  for (const auto indexed :
       llvm::enumerate(deployment.deployment().instructionCoreBinaries())) {
    auto binary = importInstructionCoreBinary(indexed.value(), artifacts, blobs);
    if (!binary)
      return binary.takeError();
    auto entry = binary->binary().threadEntry(root);
    if (!entry) {
      llvm::consumeError(entry.takeError());
      continue;
    }
    auto target = importCompilerTargetBinding(
        binary->binary().compilerTargetBinding(), artifacts);
    if (!target)
      return target.takeError();
    const CompilerProcessorArchitectureRef processor =
        CompilerProcessorArchitectureRef::instruction(
            {fabricIdentity, fabric::InstructionCoreContextRef{accCore}});
    if (llvm::Error error = requireCompilerTargetCompatibility(
            target->binding(), processor, artifacts)) {
      llvm::consumeError(std::move(error));
      continue;
    }
    if (selected)
      return invalid("more than one InstructionCore binary supports the "
                     "selected thread target");
    selected = SelectedInstructionEntry{indexed.index(), *entry};
  }
  if (!selected)
    return invalid("no InstructionCore binary supports the selected thread "
                   "target");
  return *selected;
}

std::string instructionImagePath(std::size_t ordinal) {
  return "inputs/instruction-" + std::to_string(ordinal) + ".elf";
}

llvm::Expected<Gem5SystemFactsOrUnsupported>
deriveFacts(const EvaluationRequest &request,
            const CaseArtifactResolution &resolution,
            const ArtifactStore &artifacts, const BlobStore &blobs) {
  (void)resolution;
  auto subjects = systemSubjects(request);
  if (!subjects)
    return subjects.takeError();
  if (!request.workload() || !request.runtimeInput())
    return invalid("System Request has no workload/runtime pair");
  auto systemInputs = sim::importSystemSimulationInputs(
      *request.workload(), *request.runtimeInput(), artifacts, blobs);
  if (!systemInputs)
    return systemInputs.takeError();
  if (systemInputs->deployment.reference() != subjects->first)
    return invalid("System workload names a foreign Deployment");
  auto binding = importGem5SimulationBinding(subjects->second, artifacts);
  if (!binding)
    return binding.takeError();
  if (binding->binding().fabric().artifact !=
      systemInputs->deployment.deployment().systemMapping().artifact) {
    auto mapping = mapping::importSystemMapping(
        systemInputs->deployment.deployment().systemMapping(), artifacts);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().fabricIdentity() != binding->binding().fabric().artifact)
      return invalid("gem5 binding and Deployment name different Fabric roots");
  }
  if (!isEmptySystemSurface(*systemInputs))
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  const deployment::Deployment &deployment =
      systemInputs->deployment.deployment();
  if (!deployment.spatialLaunchImage())
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  auto systemMapping =
      mapping::importSystemMapping(deployment.systemMapping(), artifacts);
  if (!systemMapping)
    return systemMapping.takeError();
  ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      systemMapping->view().dataflowIdentity()};
  auto dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto contexts = mapping::projectSystemExecutionContexts(
      *dataflowView, systemMapping->view().executionBindings());
  if (!contexts)
    return contexts.takeError();
  if (contexts->spatialDomains.size() != 1)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  const dataflow::RootedGraphLaunchRef graph =
      contexts->spatialDomains.front().graph;
  auto rootDomain =
      dataflowView->projectWholeRootedGraphLogicalDomain(graph);
  if (!rootDomain)
    return rootDomain.takeError();
  if (!*rootDomain || (*rootDomain)->coordinateRank != 0)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  auto launch = dataflowView->resolve(graph.staticGraphLaunch);
  if (!launch)
    return launch.takeError();
  auto launchOp = llvm::dyn_cast<dataflow::GraphLaunchOp>(launch->op);
  if (!launchOp || !launchOp.getValueInputs().empty() ||
      !launchOp.getStreamInputs().empty() ||
      !launchOp.getMemoryInputs().empty())
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  auto selection = deployment::resolveDeploymentSpatialLaunchSelection(
      systemInputs->deployment, graph, {}, artifacts);
  if (!selection)
    return selection.takeError();
  auto spatialMapping =
      mapping::importSpatialMapping(selection->spatialMapping, artifacts);
  if (!spatialMapping)
    return spatialMapping.takeError();
  ArtifactRootReference spatialFabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      spatialMapping->view().fabricIdentity()};

  sim::SpatialSimulationWorkload spatialWorkloadDraft{graph};
  auto spatialWorkload =
      sim::finalizeSimulationWorkload(spatialWorkloadDraft, *dataflowView);
  if (!spatialWorkload)
    return spatialWorkload.takeError();
  auto workloadReference =
      sim::publishSimulationWorkload(*spatialWorkload, artifacts);
  if (!workloadReference)
    return workloadReference.takeError();
  sim::SpatialSimulationRuntimeInputDraft spatialRuntimeDraft{
      spatialWorkload->identity()};
  auto spatialRuntime = sim::finalizeSimulationRuntimeInput(
      spatialRuntimeDraft, *spatialWorkload, *dataflowView);
  if (!spatialRuntime)
    return spatialRuntime.takeError();
  auto runtimeReference =
      sim::publishSimulationRuntimeInput(*spatialRuntime, artifacts);
  if (!runtimeReference)
    return runtimeReference.takeError();

  std::vector<Gem5ProcessorProjection> processors;
  std::optional<Gem5SpatialBridgeParameters> bridge;
  std::optional<Gem5SimpleMemoryParameters> memory;
  std::set<std::vector<std::uint8_t>> seenProcessors;
  std::set<std::vector<std::uint8_t>> seenMemories;
  std::size_t bridgeRows = 0;
  for (const Gem5Correspondence &row : binding->binding().correspondences()) {
    if (const auto *processor =
            std::get_if<Gem5ProcessorCorrespondence>(&row)) {
      if (processor->simObject.contract !=
          gem5ModelContractDescriptorRef(gem5RiscvTimingCpuModel()))
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      if (!seenProcessors.insert(processor->simObject.payload).second)
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      auto parameters =
          decodeGem5RiscvTimingCpuParameters(processor->simObject.payload);
      if (!parameters)
        return parameters.takeError();
      processors.push_back({processor->processor, *parameters});
      continue;
    }
    if (const auto *spatial =
            std::get_if<Gem5SpatialBridgeCorrespondence>(&row)) {
      ++bridgeRows;
      if (spatial->bridgeEndpoint.object.contract !=
          gem5ModelContractDescriptorRef(gem5SpatialBridgeModel()))
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      auto parameters = decodeGem5SpatialBridgeParameters(
          spatial->bridgeEndpoint.object.payload);
      if (!parameters)
        return parameters.takeError();
      if (bridge && !(*bridge == *parameters))
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      bridge = *parameters;
      continue;
    }
    if (const auto *service =
            std::get_if<Gem5MemoryOrServiceCorrespondence>(&row)) {
      if (service->simObject.contract !=
          gem5ModelContractDescriptorRef(gem5SimpleMemoryModel()))
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      if (seenMemories.insert(service->simObject.payload).second) {
        auto parameters =
            decodeGem5SimpleMemoryParameters(service->simObject.payload);
        if (!parameters)
          return parameters.takeError();
        if (memory && !(*memory == *parameters))
          return Gem5SystemFactsOrUnsupported{
              UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
        memory = *parameters;
      }
      continue;
    }
    if (const auto *transport = std::get_if<Gem5TransportCorrespondence>(&row))
      if (transport->simObject.contract !=
          gem5ModelContractDescriptorRef(gem5SystemXBarModel()))
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  }
  if (processors.empty() || !bridge || !memory || bridgeRows != 1)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  llvm::sort(processors, [](const auto &lhs, const auto &rhs) {
    return lhs.parameters.cpuId < rhs.parameters.cpuId;
  });
  if (std::adjacent_find(processors.begin(), processors.end(),
                         [](const auto &lhs, const auto &rhs) {
        return lhs.parameters.cpuId == rhs.parameters.cpuId;
      }) != processors.end() ||
      llvm::any_of(processors, [&](const auto &processor) {
        return processor.parameters.clockPeriodTicks !=
               processors.front().parameters.clockPeriodTicks;
      }))
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  const Gem5ProcessorProjection *hostProcessor = nullptr;
  const Gem5ProcessorProjection *instructionProcessor = nullptr;
  for (const Gem5ProcessorProjection &processor : processors) {
    if (std::holds_alternative<fabric::HostCoreOccurrenceRef>(
            processor.processor)) {
      if (hostProcessor)
        return invalid("gem5 binding contains more than one HostCore CPU");
      hostProcessor = &processor;
      continue;
    }
    const auto context =
        std::get<fabric::InstructionCoreContextRef>(processor.processor);
    if (context.core == selection->context.accCore) {
      if (instructionProcessor)
        return invalid("gem5 binding repeats the selected InstructionCore");
      instructionProcessor = &processor;
    }
  }
  if (!hostProcessor || !instructionProcessor)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  const std::uint64_t hostCpuId = hostProcessor->parameters.cpuId;
  const std::uint64_t instructionCpuId =
      instructionProcessor->parameters.cpuId;
  const auto *systemWorkload = systemInputs->workload.system();
  if (!systemWorkload)
    return invalid("imported System workload lost its typed payload");
  auto hostEntry = deployment::resolveDeploymentProgramEntry(
      systemInputs->deployment, systemWorkload->programEntryRef);
  if (!hostEntry)
    return hostEntry.takeError();
  auto selectedInstruction = selectInstructionEntry(
      systemInputs->deployment, graph.rootThreadLaunch,
      selection->context.accCore, systemMapping->view().fabricIdentity(),
      artifacts, blobs);
  if (!selectedInstruction)
    return selectedInstruction.takeError();

  auto memoryEndValue = checkedAdd(memory->baseAddress, memory->sizeBytes,
                                   "gem5 memory");
  auto bridgeEndValue =
      checkedAdd(bridge->pioAddress, bridge->pioSize, "Spatial Bridge");
  if (!memoryEndValue || !bridgeEndValue)
    return llvm::joinErrors(
        memoryEndValue ? llvm::Error::success() : memoryEndValue.takeError(),
        bridgeEndValue ? llvm::Error::success() : bridgeEndValue.takeError());
  const std::uint64_t memoryEnd = *memoryEndValue;
  const std::uint64_t bridgeEnd = *bridgeEndValue;
  if (memory->baseAddress < bridgeEnd && bridge->pioAddress < memoryEnd)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  auto dispatchAddressValue = alignUp(bridgeEnd, kGem5PageBytes,
                                      "Thread Dispatch");
  if (!dispatchAddressValue)
    return dispatchAddressValue.takeError();
  const std::uint64_t dispatchAddress = *dispatchAddressValue;
  auto dispatchEndValue = checkedAdd(dispatchAddress,
                                     kThreadDispatchApertureBytes,
                                     "Thread Dispatch");
  if (!dispatchEndValue)
    return dispatchEndValue.takeError();
  if (memory->baseAddress < *dispatchEndValue && dispatchAddress < memoryEnd)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  auto semanticInputs = materializeDeploymentPackage(
      systemInputs->deployment, artifacts, blobs);
  if (!semanticInputs)
    return semanticInputs.takeError();
  if (llvm::Error error =
          appendStoredObject(*semanticInputs, *workloadReference, artifacts))
    return std::move(error);
  if (llvm::Error error =
          appendStoredObject(*semanticInputs, *runtimeReference, artifacts))
    return std::move(error);
  auto hostElf = blobs.get(deployment.hostProgram().programBlob());
  if (!hostElf)
    return hostElf.takeError();
  semanticInputs->push_back({kHostElfPath.str(), bytesToString(*hostElf),
                             systemInputs->deployment.reference(), true});
  std::vector<Gem5InstructionImage> instructionImages;
  instructionImages.reserve(deployment.instructionCoreBinaries().size());
  for (const auto indexed :
       llvm::enumerate(deployment.instructionCoreBinaries())) {
    auto binary = importInstructionCoreBinary(indexed.value(), artifacts, blobs);
    if (!binary)
      return binary.takeError();
    auto bytes = blobs.get(binary->binary().codeBlob());
    if (!bytes)
      return bytes.takeError();
    const std::string path = instructionImagePath(indexed.index());
    semanticInputs->push_back(
        {path, bytesToString(*bytes), indexed.value(), true});
    instructionImages.push_back({indexed.value(), path});
  }
  const auto &launchBytes =
      deployment.spatialLaunchImage()->canonicalBytes().bytes();
  const auto &threadBytes = deployment.threadDispatchImage().canonicalBytes().bytes();
  const auto &admissionBytes = deployment.admissionImage().canonicalBytes().bytes();
  semanticInputs->push_back(
      {kSpatialLaunchPath.str(), bytesToString(launchBytes),
       systemInputs->deployment.reference(), false});
  semanticInputs->push_back(
      {kThreadDispatchPath.str(), bytesToString(threadBytes),
       systemInputs->deployment.reference(), false});
  semanticInputs->push_back(
      {kAdmissionPath.str(), bytesToString(admissionBytes),
       systemInputs->deployment.reference(), false});

  auto midpointValue = checkedAdd(memory->baseAddress, memory->sizeBytes / 2,
                                  "gem5 runtime image arena");
  if (!midpointValue)
    return midpointValue.takeError();
  auto cursorValue = alignUp(*midpointValue, kGem5PageBytes,
                             "gem5 runtime image arena");
  if (!cursorValue)
    return cursorValue.takeError();
  std::uint64_t cursor = *cursorValue;
  std::vector<Gem5RuntimeImage> runtimeImages;
  auto placeRuntimeImage = [&](llvm::StringRef path,
                               std::uint64_t size)
      -> llvm::Expected<std::uint64_t> {
    if (size == 0)
      return invalid("gem5 runtime image is empty");
    const std::uint64_t address = cursor;
    auto end = checkedAdd(cursor, size, "gem5 runtime image");
    if (!end)
      return end.takeError();
    auto aligned = alignUp(*end, kGem5PageBytes, "gem5 runtime image");
    if (!aligned)
      return aligned.takeError();
    cursor = *aligned;
    runtimeImages.push_back({path.str(), address});
    return address;
  };
  auto threadAddress =
      placeRuntimeImage(kThreadDispatchPath, threadBytes.size());
  auto admissionAddress = placeRuntimeImage(kAdmissionPath, admissionBytes.size());
  auto launchAddress = placeRuntimeImage(kSpatialLaunchPath, launchBytes.size());
  if (!threadAddress || !admissionAddress || !launchAddress)
    return llvm::joinErrors(
        threadAddress ? llvm::Error::success() : threadAddress.takeError(),
        llvm::joinErrors(
            admissionAddress ? llvm::Error::success()
                             : admissionAddress.takeError(),
            launchAddress ? llvm::Error::success() : launchAddress.takeError()));
  const std::uint64_t stackBase = cursor;
  std::uint64_t maximumCpuId = 0;
  for (const Gem5ProcessorProjection &processor : processors)
    maximumCpuId = std::max(maximumCpuId, processor.parameters.cpuId);
  auto stackCount = checkedAdd(maximumCpuId, 1, "gem5 stack count");
  if (!stackCount ||
      *stackCount > std::numeric_limits<std::uint64_t>::max() /
                        kGem5StackBytes)
    return invalid("gem5 stack arena size overflows uint64");
  auto stackEnd = checkedAdd(stackBase, *stackCount * kGem5StackBytes,
                             "gem5 stack arena");
  if (!stackEnd)
    return stackEnd.takeError();
  if (*stackEnd > memoryEnd)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  llvm::sort(*semanticInputs, [](const auto &lhs, const auto &rhs) {
    return lhs.relativePath < rhs.relativePath;
  });

  return Gem5SystemFactsOrUnsupported{Gem5SystemFacts{
      selectedEngine(request),
      systemInputs->deployment.reference(),
      binding->reference(),
      std::move(dataflowReference),
      std::move(spatialFabricReference),
      selection->spatialMapping,
      selection->hardwareImplementation,
      std::move(*workloadReference),
      std::move(*runtimeReference),
      std::vector<std::uint8_t>(launchBytes.begin(), launchBytes.end()),
      std::move(*semanticInputs),
      std::move(processors),
      (*hostEntry)->abiSymbol,
      hostCpuId,
      std::move(instructionImages),
      std::move(runtimeImages),
      dispatchAddress,
      stackBase,
      kGem5StackBytes,
      {{instructionCpuId,
        selectedInstruction->imageOrdinal,
        "__loom_thread_entry_" +
            std::to_string(selectedInstruction->entryOrdinal),
        bridge->pioAddress,
        *launchAddress,
        static_cast<std::uint64_t>(launchBytes.size())}},
      *bridge,
      *memory}};
}

llvm::Expected<std::filesystem::path>
readinessPath(const LocalToolConfig &config,
              const ExternalToolProviderDescriptor &provider,
              const ResolvedToolBinding &tool) {
  const auto configured = config.tools.find(provider.binding.key);
  if (configured != config.tools.end()) {
    if (const llvm::json::Value *value =
            configured->second.providerOptions.get("readiness")) {
      const auto path = value->getAsString();
      if (!path)
        return invalid("gem5.provider_options.readiness must be a string");
      std::filesystem::path configuredPath(path->str());
      if (!configuredPath.is_absolute() ||
          configuredPath.lexically_normal() != configuredPath)
        return invalid("gem5 readiness path must be absolute and canonical");
      return configuredPath;
    }
  }
  const std::filesystem::path executable(tool.executable);
  if (!executable.is_absolute())
    return invalid("resolved gem5 executable path is not absolute");
  const std::filesystem::path root =
      executable.parent_path().parent_path().parent_path();
  return root / "loom-gem5-readiness.json";
}

llvm::Expected<ReadinessIdentity>
verifyReadiness(const Gem5SystemFacts &facts,
                const FinalizedGem5SimulationBinding &binding,
                const LocalToolConfig &config,
                const ExternalToolProviderDescriptor &provider,
                const ResolvedToolBinding &tool) {
  auto path = readinessPath(config, provider, tool);
  if (!path)
    return path.takeError();
  auto contents = readFile(path->string());
  if (!contents)
    return contents.takeError();
  auto value = llvm::json::parse(*contents);
  if (!value)
    return invalid("gem5 readiness stamp is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return invalid("gem5 readiness stamp is not an object");
  const auto schema = object->getString("schema");
  const auto bridgeAbi = object->getString("bridge_abi_identity");
  const auto repository = object->getString("gem5_repository_identity");
  const auto commit = object->getString("gem5_full_commit_identity");
  const auto configuration = object->getString("build_configuration_digest");
  const auto binary = object->getString("binary");
  const auto binarySha = object->getString("binary_sha256");
  const auto versionProbe = object->getString("version_probe");
  if (!schema || !bridgeAbi || !repository || !commit || !configuration ||
      !binary || !binarySha || !versionProbe)
    return invalid("gem5 readiness stamp omits an identity field");
  const Gem5BuildIdentity &expected =
      binding.binding().gem5BuildIdentity();
  if (*schema != "loom.gem5_build_readiness.1" ||
      *bridgeAbi != binding.binding().bridgeAbiIdentity() ||
      *repository != expected.repositoryIdentity ||
      *commit != expected.fullCommitIdentity ||
      *configuration != expected.buildConfigurationDigest ||
      *binarySha != expected.binaryFingerprint)
    return invalid("gem5 readiness identity differs from the exact binding");
  std::error_code error;
  const std::filesystem::path resolvedTool =
      std::filesystem::weakly_canonical(tool.executable, error);
  if (error)
    return invalid("cannot canonicalize the resolved gem5 executable");
  const std::filesystem::path recordedBinary(binary->str());
  if (recordedBinary != resolvedTool || !versionProbe->contains(tool.version))
    return invalid("gem5 readiness does not describe the resolved executable");
  auto binaryContents = readFile(tool.executable);
  if (!binaryContents)
    return binaryContents.takeError();
  const BlobDigest digest = computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(binaryContents->data()),
      binaryContents->size()));
  if (formatBlobDigestHex(digest) != *binarySha)
    return invalid("resolved gem5 executable differs from its readiness stamp");
  if (facts.launchPayload.size() > facts.bridge.maximumMessageBytes)
    return invalid("Deployment launch image exceeds the bridge message limit");
  auto fingerprint = parseExternalFileFingerprint(*binarySha);
  if (!fingerprint)
    return fingerprint.takeError();
  return ReadinessIdentity{binarySha->str(), std::move(*fingerprint)};
}

std::vector<std::string> inheritedEnvironment(
    const LocalToolConfig &config,
    const ExternalToolProviderDescriptor &provider) {
  const auto configured = config.tools.find(provider.binding.key);
  if (configured == config.tools.end())
    return {};
  return configured->second.inheritEnvironment;
}

std::string renderProjection(const Gem5SystemFacts &facts,
                             const ReadinessIdentity &readiness) {
  std::string output;
  llvm::raw_string_ostream stream(output);
  llvm::json::OStream json(stream, 0);
  const std::uint64_t ticksPerCycle =
      facts.processors.front().parameters.clockPeriodTicks;
  const std::string engine = facts.engine == Gem5SystemEngine::Dfg
                                 ? kDfgEnginePath.str()
                                 : kCgraEnginePath.str();
  json.object([&] {
    json.attribute("schema", "loom.gem5_system_projection.2");
    json.attribute("gem5_binary_sha256", readiness.binarySha256);
    json.attribute("clock", std::to_string(ticksPerCycle) + "ps");
    json.attributeObject("memory", [&] {
      json.attribute("base", facts.memory.baseAddress);
      json.attribute("size", facts.memory.sizeBytes);
      json.attribute("latency",
                     std::to_string(facts.memory.latencyTicks) + "ps");
    });
    json.attributeObject("host", [&] {
      json.attribute("elf", kHostElfPath);
      json.attribute("cpu_id", facts.hostCpuId);
      json.attribute("entry_symbol", facts.hostEntrySymbol);
    });
    json.attributeArray("instruction_images", [&] {
      for (const Gem5InstructionImage &image : facts.instructionImages)
        json.value(image.path);
    });
    json.attributeArray("runtime_images", [&] {
      for (const Gem5RuntimeImage &image : facts.runtimeImages)
        json.object([&] {
          json.attribute("path", image.path);
          json.attribute("address", image.address);
        });
    });
    json.attributeObject("dispatch", [&] {
      json.attribute("pio_address", facts.dispatchAddress);
      json.attribute("pio_latency", std::to_string(ticksPerCycle) + "ps");
      json.attribute("stack_base", facts.stackBase);
      json.attribute("stack_stride", facts.stackStride);
      json.attributeArray("targets", [&] {
        for (const Gem5DispatchTarget &target : facts.dispatchTargets)
          json.object([&] {
            json.attribute("cpu_id", target.cpuId);
            json.attribute("image_ordinal", target.imageOrdinal);
            json.attribute("entry_symbol", target.entrySymbol);
            json.attribute("bridge_address", target.bridgeAddress);
            json.attribute("launch_address", target.launchAddress);
            json.attribute("launch_size", target.launchSize);
          });
      });
    });
    json.attributeArray("processors", [&] {
      for (const Gem5ProcessorProjection &processor : facts.processors)
        json.object([&] {
          json.attribute("cpu_id", processor.parameters.cpuId);
        });
    });
    json.attributeArray("bridges", [&] {
      json.object([&] {
        json.attribute("pio_address", facts.bridge.pioAddress);
        json.attribute("pio_size", facts.bridge.pioSize);
        json.attribute("pio_latency",
                       std::to_string(facts.bridge.pioLatencyTicks) + "ps");
        json.attribute("engine_socket", "outputs/spatial-bridge-0.sock");
        json.attributeArray("engine_command", [&] {
          if (facts.engine == Gem5SystemEngine::Rtl)
            return;
          json.value(engine);
          json.value("--artifact-store");
          json.value(kPackageObjectPath);
          json.value("--socket");
          json.value("outputs/spatial-bridge-0.sock");
          json.value("--expected-launch");
          json.value(kSpatialLaunchPath);
          json.value("--workload");
          json.value(formatArtifactIdentityHex(facts.spatialWorkload.artifact));
          json.value("--runtime-input");
          json.value(
              formatArtifactIdentityHex(facts.spatialRuntimeInput.artifact));
          json.value("--dataflow");
          json.value(formatArtifactIdentityHex(facts.dataflow.artifact));
          json.value("--maximum-work");
          json.value(std::to_string(kMaximumSpatialWork));
          json.value("--ticks-per-cycle");
          json.value(std::to_string(ticksPerCycle));
          if (facts.engine == Gem5SystemEngine::Cgra) {
            json.value("--fabric");
            json.value(formatArtifactIdentityHex(facts.fabric.artifact));
            json.value("--spatial-mapping");
            json.value(
                formatArtifactIdentityHex(facts.spatialMapping.artifact));
          }
        });
        json.attribute("result_path", kBridgeResultPath);
        json.attribute("maximum_message_bytes",
                       facts.bridge.maximumMessageBytes);
      });
    });
    json.attribute("maximum_ticks", kMaximumGem5Ticks);
  });
  stream << '\n';
  stream.flush();
  return output;
}

ExternalToolInvocationImportExpectation
makeExpectation(const ExternalToolSemanticContract &contract,
                const Gem5SystemFacts &facts,
                llvm::ArrayRef<ExternalToolInvocationSemanticInput>
                    additionalSemanticInputs = {},
                std::optional<ExternalFileFingerprint> gem5Binary =
                    std::nullopt) {
  ExternalToolInvocationImportExpectation expectation;
  expectation.semanticContract = contract;
  for (const MaterializedBundleFile &file : facts.semanticInputs) {
    if (!file.sourceArtifact)
      continue;
    expectation.semanticInputs.push_back({
        file.relativePath, *file.sourceArtifact,
        computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(file.contents.data()),
            file.contents.size()))});
  }
  expectation.semanticInputs.insert(expectation.semanticInputs.end(),
                                    additionalSemanticInputs.begin(),
                                    additionalSemanticInputs.end());
  llvm::sort(expectation.semanticInputs, [](const auto &lhs, const auto &rhs) {
    return lhs.relativePath < rhs.relativePath;
  });
  if (gem5Binary)
    expectation.externalInputs.push_back(
        {"gem5_binary", std::move(*gem5Binary)});
  expectation.declaredOutputs = {kSystemResultPath.str(),
                                 kBridgeResultPath.str()};
  if (facts.engine == Gem5SystemEngine::Rtl)
    expectation.declaredOutputs.push_back(
        eda::open_source::mappedRtlResultPath.str());
  return expectation;
}

llvm::Expected<ExternalFileFingerprint> gem5BinaryFingerprint(
    const FinalizedGem5SimulationBinding &binding) {
  return parseExternalFileFingerprint(
      binding.binding().gem5BuildIdentity().binaryFingerprint);
}

llvm::Expected<eda::open_source::MappedRtlExecutionClosure>
mappedRtlClosure(const EvaluationRequest &request,
                 const Gem5SystemFacts &facts,
                 const ExternalToolSemanticContract &contract) {
  const auto *binding = request.modelBinding()
                            .resolvedModelConfig()
                            .getIf<models::MappedRtlSimulatorBinding>();
  if (!binding)
    return invalid("gem5 RTL Request has no HDL simulator binding");
  if (llvm::Error error = models::validateMappedRtlSimulatorBinding(*binding))
    return std::move(error);
  return eda::open_source::MappedRtlExecutionClosure{
      *binding,
      contract,
      facts.hardwareImplementation,
      facts.deployment,
      facts.spatialWorkload,
      facts.spatialRuntimeInput};
}

EvaluationModelResult terminalResult(EvaluationEvidenceOutcome outcome) {
  return EvaluationModelResult{{{kExecutionOutput, {}}}, std::move(outcome)};
}

llvm::Expected<EvaluationModelResult>
classifyFailedAttempt(const FailedExternalToolInvocationAttempt &failed) {
  switch (failed.status) {
  case InvocationCompletionStatus::Success:
    return invalid("failed gem5 attempt carries success status");
  case InvocationCompletionStatus::MissingEnvironment:
  case InvocationCompletionStatus::ModuleActivationFailed:
  case InvocationCompletionStatus::VersionMismatch:
    return terminalResult(
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable});
  case InvocationCompletionStatus::BundleContentMismatch:
    return invalid("gem5 invocation bundle changed before execution");
  case InvocationCompletionStatus::ToolExit:
  case InvocationCompletionStatus::MissingOutput:
    return terminalResult(ExecutionFailedEvidence{OutcomeReason::ToolFailure});
  }
  llvm_unreachable("closed invocation status");
}

struct Gem5AttemptResult final {
  std::uint64_t entryTick = 0;
  std::uint64_t exitTick = 0;
  std::string cause;
};

llvm::Expected<Gem5AttemptResult> parseAttemptResult(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return invalid("gem5 result is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object || object->size() != 4)
    return invalid("gem5 result does not have the exact result shape");
  const auto schema = object->getString("schema");
  const auto entry = object->getInteger("entry_tick");
  const auto exit = object->getInteger("exit_tick");
  const auto cause = object->getString("cause");
  if (!schema || *schema != "loom.gem5_system_attempt.1" || !entry || !exit ||
      !cause || *entry < 0 || *exit < 0 || *entry > *exit)
    return invalid("gem5 result fields are invalid");
  return Gem5AttemptResult{static_cast<std::uint64_t>(*entry),
                           static_cast<std::uint64_t>(*exit), cause->str()};
}

} // namespace

llvm::Expected<EvaluationModelProviderPreparation>
prepareGem5SystemInvocation(
    const EvaluationRequest &request,
    const CaseArtifactResolution &resolution, const ArtifactStore &artifacts,
    const BlobStore &blobs, const ExternalToolPreparationContext &context) {
  auto factsOrUnsupported = deriveFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<UnsupportedEvidence>(&*factsOrUnsupported))
    return EvaluationModelProviderPreparation{*unsupported};
  Gem5SystemFacts facts =
      std::get<Gem5SystemFacts>(std::move(*factsOrUnsupported));

  auto subjects = systemSubjects(request);
  if (!subjects)
    return subjects.takeError();
  auto binding = importGem5SimulationBinding(subjects->second, artifacts);
  if (!binding)
    return binding.takeError();
  const ExternalToolProviderDescriptor &gem5ToolProvider = gem5Provider();
  const std::filesystem::path destination(context.bundleDestination);
  const std::filesystem::path probeRoot = destination.parent_path();
  ShellToolBindingProbe gem5Probe(probeRoot.string(),
                                  gem5ToolProvider.versionProbe);
  auto gem5Tool = resolveToolBinding(
      gem5ToolProvider.binding, context.localConfig,
      captureToolEnvironment(gem5ToolProvider.binding), gem5Probe);
  if (!gem5Tool)
    return gem5Tool.takeError();
  auto readiness = verifyReadiness(facts, *binding, context.localConfig,
                                   gem5ToolProvider, *gem5Tool);
  if (!readiness)
    return readiness.takeError();
  const std::string gem5Executable = gem5Tool->executable;
  const ResolvedExternalFile gem5ExternalFile{
      "gem5_binary", "gem5_readiness", gem5Executable,
      readiness->binaryFingerprint};

  const ExternalToolProviderDescriptor &container =
      polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeRoot.string(),
                                       container.versionProbe);
  auto contract = deriveExternalToolSemanticContract(request);
  if (!contract)
    return contract.takeError();
  auto configuration = readFile(LOOM_GEM5_CONFIG_SCRIPT_PATH);
  if (!configuration)
    return configuration.takeError();
  std::vector<MaterializedBundleFile> files = std::move(facts.semanticInputs);
  files.push_back({kConfigurationScriptPath.str(), std::move(*configuration),
                   std::nullopt, false});
  files.push_back({kProjectionPath.str(), renderProjection(facts, *readiness),
                   std::nullopt, false});

  if (facts.engine == Gem5SystemEngine::Rtl) {
    auto closure = mappedRtlClosure(request, facts, *contract);
    if (!closure)
      return closure.takeError();
    auto options = eda::open_source::resolveMappedRtlExecutionAttemptOptions(
        context.localConfig);
    if (!options)
      return options.takeError();
    const ExternalToolProviderDescriptor &verilatorToolProvider =
        verilatorProvider();
    ShellToolBindingProbe verilatorProbe(probeRoot.string(),
                                         verilatorToolProvider.versionProbe);
    auto verilatorTool = resolveToolBinding(
        verilatorToolProvider.binding, context.localConfig,
        captureToolEnvironment(verilatorToolProvider.binding), verilatorProbe);
    if (!verilatorTool)
      return verilatorTool.takeError();
    if (verilatorTool->version !=
        closure->simulatorBinding.stableHdlSimulatorBuildIdentity)
      return invalid(
          "resolved Verilator build differs from the gem5 RTL binding");
    const std::string verilatorExecutable = verilatorTool->executable;
    auto runtime = resolveInvocationRuntime(
        *verilatorTool, context.localConfig, container.binding,
        captureToolEnvironment(container.binding), containerProbe,
        verilatorToolProvider.runtimeCompatibility,
        [&](const ResolvedToolBinding &resolvedTool,
            const ResolvedToolBinding &resolvedContainer,
            llvm::StringRef os)
            -> llvm::Expected<std::optional<std::string>> {
          return probeContainerToolComposition(
              probeRoot.string(), resolvedTool,
              verilatorToolProvider.versionProbe, resolvedContainer, os,
              options->inheritedEnvironment);
        });
    if (!runtime)
      return runtime.takeError();
    auto projection =
        eda::open_source::deriveMappedRtlExecutionBundleProjection(
            *closure, options->cycleLimit, options->buildJobs, artifacts,
            blobs);
    if (!projection)
      return projection.takeError();
    if (const auto *unsupported =
            std::get_if<UnsupportedEvidence>(&*projection))
      return EvaluationModelProviderPreparation{*unsupported};
    auto rtl = std::get<
        eda::open_source::MappedRtlExecutionBundleProjection>(
        std::move(*projection));
    auto engineSource = readFile(LOOM_GEM5_RTL_ENGINE_SOURCE_PATH);
    auto bridgeHeader = readFile(LOOM_GEM5_BRIDGE_HEADER_PATH);
    if (!engineSource || !bridgeHeader)
      return llvm::joinErrors(
          engineSource ? llvm::Error::success() : engineSource.takeError(),
          bridgeHeader ? llvm::Error::success() : bridgeHeader.takeError());
    files.push_back({eda::open_source::mappedRtlTestbenchPath.str(),
                     std::move(rtl.testbench), std::nullopt, false});
    files.push_back(
        {eda::open_source::mappedRtlBridgedVerilatorDriverPath.str(),
         std::move(rtl.bridgedVerilatorDriver), std::nullopt, false});
    files.push_back({eda::open_source::mappedRtlBridgeEngineSourcePath.str(),
                     std::move(*engineSource), std::nullopt, false});
    files.push_back({kBridgeHeaderPath.str(), std::move(*bridgeHeader),
                     std::nullopt, false});
    files.insert(files.end(),
                 std::make_move_iterator(rtl.semanticInputs.begin()),
                 std::make_move_iterator(rtl.semanticInputs.end()));
    std::vector<std::string> engineCommand{
        eda::open_source::mappedRtlSimulatorExecutablePath.str(),
        "--socket",
        "outputs/spatial-bridge-0.sock",
        "--expected-launch",
        kSpatialLaunchPath.str(),
        "--mapped-result",
        eda::open_source::mappedRtlResultPath.str(),
        "--ticks-per-cycle",
        std::to_string(
            facts.processors.front().parameters.clockPeriodTicks),
        "--gem5",
        gem5Executable,
        "--gem5-output",
        "outputs/gem5",
        "--gem5-config",
        kConfigurationScriptPath.str(),
        "--projection",
        kProjectionPath.str(),
        "--system-result",
        kSystemResultPath.str()};
    if (options->debugVerbosity != 0)
      engineCommand.push_back("+LOOM_DEBUG_VERBOSE=" +
                              std::to_string(options->debugVerbosity));
    ExternalToolInvocationBundleSpec specification{
        std::move(*contract),
        std::move(*verilatorTool),
        verilatorToolProvider.versionProbe,
        std::move(*runtime),
        container.versionProbe,
        {{verilatorExecutable,
          "-f",
          eda::open_source::mappedRtlBridgedVerilatorDriverPath.str()},
         std::move(engineCommand)},
        std::move(options->inheritedEnvironment),
        {kSystemResultPath.str(), kBridgeResultPath.str(),
         eda::open_source::mappedRtlResultPath.str()},
        std::move(files),
        {gem5ExternalFile},
        {},
        {eda::open_source::mappedRtlSimulatorExecutablePath.str()}};
    llvm::sort(specification.files, [](const auto &lhs, const auto &rhs) {
      return lhs.relativePath < rhs.relativePath;
    });
    auto prepared = finalizeExternalToolInvocationBundle(
        context.bundleDestination, specification);
    if (!prepared)
      return prepared.takeError();
    return EvaluationModelProviderPreparation{std::move(*prepared)};
  }

  const std::vector<std::string> inherited =
      inheritedEnvironment(context.localConfig, gem5ToolProvider);
  auto runtime = resolveInvocationRuntime(
      *gem5Tool, context.localConfig, container.binding,
      captureToolEnvironment(container.binding), containerProbe,
      gem5ToolProvider.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &resolvedContainer,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        return probeContainerToolComposition(
            probeRoot.string(), resolvedTool, gem5ToolProvider.versionProbe,
            resolvedContainer, os, inherited);
      });
  if (!runtime)
    return runtime.takeError();
  const llvm::StringRef engineSource =
      facts.engine == Gem5SystemEngine::Dfg ? LOOM_GEM5_DFG_ENGINE_PATH
                                            : LOOM_GEM5_CGRA_ENGINE_PATH;
  auto engine = readFile(engineSource);
  if (!engine)
    return engine.takeError();
  files.push_back({facts.engine == Gem5SystemEngine::Dfg ? kDfgEnginePath.str()
                                                         : kCgraEnginePath.str(),
                   std::move(*engine), std::nullopt, true});

  ExternalToolInvocationBundleSpec specification{
      std::move(*contract),
      std::move(*gem5Tool),
      gem5ToolProvider.versionProbe,
      std::move(*runtime),
      container.versionProbe,
      {},
      inherited,
      {kSystemResultPath.str(), kBridgeResultPath.str()},
      std::move(files),
      {gem5ExternalFile},
      {},
      {}};
  specification.commands = {{specification.tool.executable,
                             "-d",
                             "outputs/gem5",
                             kConfigurationScriptPath.str(),
                             "--projection",
                             kProjectionPath.str(),
                             "--result",
                             kSystemResultPath.str()}};
  auto prepared = finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
  if (!prepared)
    return prepared.takeError();
  return EvaluationModelProviderPreparation{std::move(*prepared)};
}

llvm::Expected<EvaluationModelResult>
importGem5SystemInvocation(
    const EvaluationRequest &request,
    const CaseArtifactResolution &resolution,
    const PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto factsOrUnsupported = deriveFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (std::holds_alternative<UnsupportedEvidence>(*factsOrUnsupported))
    return invalid("prepared gem5 invocation is outside provider capability");
  Gem5SystemFacts facts =
      std::get<Gem5SystemFacts>(std::move(*factsOrUnsupported));
  auto contract = deriveExternalToolSemanticContract(request);
  if (!contract)
    return contract.takeError();
  auto subjects = systemSubjects(request);
  if (!subjects)
    return subjects.takeError();
  auto binding = importGem5SimulationBinding(subjects->second, artifacts);
  if (!binding)
    return binding.takeError();
  auto fingerprint = gem5BinaryFingerprint(*binding);
  if (!fingerprint)
    return fingerprint.takeError();
  std::vector<ExternalToolInvocationSemanticInput> mappedRtlInputs;
  std::optional<eda::open_source::MappedRtlExecutionClosure> rtlClosure;
  if (facts.engine == Gem5SystemEngine::Rtl) {
    auto closure = mappedRtlClosure(request, facts, *contract);
    if (!closure)
      return closure.takeError();
    auto expectation =
        eda::open_source::deriveMappedRtlExecutionImportExpectation(
            *closure, artifacts, blobs);
    if (!expectation)
      return expectation.takeError();
    mappedRtlInputs = std::move(expectation->semanticInputs);
    rtlClosure = std::move(*closure);
  }
  auto attempt = importExternalToolInvocationAttempt(
      prepared,
      makeExpectation(*contract, facts, mappedRtlInputs, *fingerprint));
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<FailedExternalToolInvocationAttempt>(&*attempt))
    return classifyFailedAttempt(*failed);
  ImportedExternalToolInvocationBundle imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  auto systemText =
      readExternalToolInvocationDeclaredOutput(imported, kSystemResultPath);
  if (!systemText)
    return systemText.takeError();
  auto systemResult = parseAttemptResult(*systemText);
  if (!systemResult)
    return systemResult.takeError();
  if (!llvm::StringRef(systemResult->cause).contains("m5_exit"))
    return terminalResult(
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
  auto bridgeText =
      readExternalToolInvocationDeclaredOutput(imported, kBridgeResultPath);
  if (!bridgeText)
    return bridgeText.takeError();
  std::vector<std::uint8_t> bridgeBytes(bridgeText->begin(), bridgeText->end());
  Gem5BridgeResult bridgeResult;
  std::string bridgeDiagnostic;
  if (!decodeGem5BridgeResult(bridgeBytes, bridgeResult, bridgeDiagnostic))
    return invalid("bridge result is invalid: " + bridgeDiagnostic);
  if (bridgeResult.status > 1 || bridgeResult.sequence != 0 ||
      bridgeResult.completionTick < systemResult->entryTick ||
      bridgeResult.completionTick > systemResult->exitTick)
    return invalid("bridge completion is inconsistent with gem5 time");
  std::optional<sim::SpatialEngineBoundaryResult> spatialResult;
  if (facts.engine == Gem5SystemEngine::Rtl) {
    if (!rtlClosure)
      return invalid("gem5 RTL import lost its exact mapped RTL closure");
    auto mappedText = readExternalToolInvocationDeclaredOutput(
        imported, eda::open_source::mappedRtlResultPath);
    if (!mappedText)
      return mappedText.takeError();
    const llvm::ArrayRef<std::uint8_t> mappedBytes(
        reinterpret_cast<const std::uint8_t *>(mappedText->data()),
        mappedText->size());
    if (mappedBytes != llvm::ArrayRef<std::uint8_t>(bridgeResult.result))
      return invalid("bridge payload differs from the mapped RTL result");
    auto mappedResult =
        eda::open_source::parseMappedRtlSimulationResult(*mappedText);
    if (!mappedResult)
      return mappedResult.takeError();
    if (mappedResult->terminal ==
        eda::open_source::MappedRtlTerminalStatus::StoppedByLimit) {
      if (bridgeResult.status != 1)
        return invalid("bridge status disagrees with the RTL terminal");
      return terminalResult(
          CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
    }
    if (bridgeResult.status != 0)
      return invalid("bridge status disagrees with the RTL terminal");
    auto boundary =
        eda::open_source::projectMappedRtlSpatialEngineBoundaryResult(
            *rtlClosure, *mappedResult, artifacts, blobs);
    if (!boundary)
      return boundary.takeError();
    spatialResult = std::move(*boundary);
  } else {
    auto boundary = sim::decodeSpatialEngineBoundaryResult(
        bridgeResult.result, facts.spatialWorkload,
        facts.spatialRuntimeInput, artifacts);
    if (!boundary)
      return boundary.takeError();
    const std::uint32_t expectedStatus =
        std::holds_alternative<sim::RetiredExecution>(boundary->terminal) ? 0U
                                                                         : 1U;
    if (bridgeResult.status != expectedStatus)
      return invalid("bridge status disagrees with the Spatial terminal");
    spatialResult = std::move(*boundary);
  }
  if (!spatialResult ||
      !std::holds_alternative<sim::RetiredExecution>(spatialResult->terminal))
    return terminalResult(
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});

  sim::SystemSimulationExecution execution{
      evaluationRequestReference(request),
      sim::RetiredExecution{},
      {},
      {{systemResult->entryTick, 0},
       sim::SystemEventCoordinate{systemResult->exitTick, 0},
       {systemResult->exitTick, 0}},
      {}};
  auto finalized =
      sim::finalizeSimulationExecution(execution, resolution, artifacts, blobs);
  if (!finalized)
    return finalized.takeError();
  auto executionReference =
      sim::publishSimulationExecution(*finalized, artifacts);
  if (!executionReference)
    return executionReference.takeError();

  const std::uint64_t duration =
      systemResult->exitTick - systemResult->entryTick;
  if (duration >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return terminalResult(
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure});
  std::vector<MetricResult> metrics;
  metrics.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::Runtime)
      return invalid("gem5 System Request contains an unsupported metric");
    auto runtime = DecimalValue::get(static_cast<std::int64_t>(duration), -12);
    if (!runtime)
      return runtime.takeError();
    metrics.push_back(
        {UncertaintyKind::ExactWithinModel,
         PointObservation{MetricValue(std::move(*runtime))}, {}});
  }
  return EvaluationModelResult{
      {{kExecutionOutput, {std::move(*executionReference)}}},
      CompletedEvidence{std::move(metrics), {}}};
}

} // namespace loom::runtime
