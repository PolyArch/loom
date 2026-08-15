#include "Gem5SystemExecutionInternal.h"

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
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Runtime/Gem5SpatialChannel.h"
#include "Runtime/Gem5SpatialChannelPlan.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Matchers.h"

#include <algorithm>
#include <array>
#include <cassert>
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

namespace loom::runtime::gem5_system {

using namespace evaluation;
using namespace external_tool;

struct PendingChannelInput final {
  std::size_t producerLaunch = 0;
  std::uint64_t producerStreamOutputOrdinal = 0;
  std::uint64_t consumerStreamInputOrdinal = 0;
  std::size_t buffer = 0;
};

struct PendingChannelOutput final {
  std::uint64_t producerStreamOutputOrdinal = 0;
  std::size_t buffer = 0;
};

struct PendingSpatialLaunch final {
  dataflow::RootedGraphLaunchRef graph;
  std::vector<std::uint64_t> denseCoordinates;
  fabric::AccCoreOccurrenceRef accCore;
  mapping::SpatialExecutionContextKey context;
  std::vector<mapping::SystemPresburgerCell> contextDomain;
  ArtifactRootReference fabric;
  ArtifactRootReference spatialMapping;
  ArtifactRootReference hardwareImplementation;
  std::optional<ArtifactRootReference> spatialWorkload;
  std::optional<ArtifactRootReference> spatialRuntimeInput;
  std::vector<PendingChannelInput> channelInputs;
  std::vector<PendingChannelOutput> channelOutputs;
  std::vector<std::uint32_t> streamInputBitWidths;
  std::vector<std::uint32_t> streamOutputBitWidths;
  std::vector<std::uint64_t> observableStreamOutputOrdinals;
};

struct PendingChannelBuffer final {
  std::size_t producerLaunch = 0;
  std::uint64_t producerStreamOutputOrdinal = 0;
  std::uint64_t address = 0;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_system_execution_invalid: " + message);
}

std::string bytesToString(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

Gem5SystemEngine selectedEngine(const EvaluationRequest &request) {
  const EvaluationModelKind kind =
      request.modelBinding().descriptorRef().modelKind();
  if (kind == builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemDfg))
    return Gem5SystemEngine::Dfg;
  if (kind ==
      builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemCgra))
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

bool supportsSystemMemorySurface(
    const sim::ImportedSystemSimulationInputs &inputs) {
  const sim::SystemSimulationWorkload &workload = *inputs.workload.system();
  const sim::SystemSimulationRuntimeInput &runtime =
      *inputs.runtimeInput.system();
  return workload.valueInputPlan.empty() &&
         workload.externalValueInputPlan.empty() &&
         workload.observableContract.valueResults.empty() &&
         workload.observableContract.externalValueOutputs.empty() &&
         workload.observableContract.externalStreamOutputs.empty() &&
         runtime.runtimeEntryValues.empty() &&
         runtime.runtimeExternalValues.empty() &&
         runtime.externalStreamInputs.empty();
}

void appendGuestU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (unsigned byte = 0; byte != 4; ++byte)
    bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

void appendGuestU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte)
    bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

const sim::SystemMemoryInterfaceBindingEntry *
findMemoryBinding(const sim::SystemSimulationRuntimeInput &runtime,
                  const deployment::DeploymentExternalInterfaceRef &reference) {
  auto found = std::lower_bound(
      runtime.memoryInterfaceBindings.begin(),
      runtime.memoryInterfaceBindings.end(), reference,
      [](const sim::SystemMemoryInterfaceBindingEntry &entry,
         const deployment::DeploymentExternalInterfaceRef &target) {
        return deployment::deploymentExternalInterfaceRefLess(
            entry.interfaceRef, target);
      });
  if (found == runtime.memoryInterfaceBindings.end() ||
      !(found->interfaceRef == reference))
    return nullptr;
  return &*found;
}

llvm::Expected<std::vector<MaterializedBundleFile>>
materializeDeploymentPackage(const deployment::FinalizedDeployment &deployment,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  auto closure =
      deployment::deriveDeploymentPackageClosure(deployment, artifacts, blobs);
  if (!closure)
    return closure.takeError();
  std::vector<MaterializedBundleFile> files;
  files.reserve(closure->artifacts().size() + closure->blobs().size() + 1);
  files.push_back({"inputs/package/root",
                   formatArtifactIdentityHex(deployment.reference().artifact),
                   deployment.reference(), false});
  for (const ArtifactRootReference &reference : closure->artifacts()) {
    auto contents = artifacts.getStoredObject(reference);
    if (!contents)
      return contents.takeError();
    files.push_back({"inputs/package/objects/" +
                         formatArtifactIdentityHex(reference.artifact),
                     bytesToString(*contents), reference, false});
  }
  for (const BlobDigest &digest : closure->blobs()) {
    auto contents = blobs.get(digest);
    if (!contents)
      return contents.takeError();
    files.push_back({"inputs/package/blobs/" + formatBlobDigestHex(digest),
                     bytesToString(*contents), deployment.reference(), false});
  }
  return files;
}

llvm::Error appendStoredObject(std::vector<MaterializedBundleFile> &files,
                               const ArtifactRootReference &reference,
                               const ArtifactStore &artifacts) {
  const std::string path =
      "inputs/package/objects/" + formatArtifactIdentityHex(reference.artifact);
  if (llvm::any_of(files, [&](const MaterializedBundleFile &file) {
        return file.relativePath == path;
      }))
    return llvm::Error::success();
  auto object = artifacts.getStoredObject(reference);
  if (!object)
    return object.takeError();
  files.push_back({path, bytesToString(*object), reference, false});
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> checkedAdd(std::uint64_t lhs, std::uint64_t rhs,
                                         llvm::StringRef role) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    return invalid(role + " address range overflows uint64");
  return lhs + rhs;
}

llvm::Expected<std::uint64_t>
alignUp(std::uint64_t value, std::uint64_t alignment, llvm::StringRef role) {
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

llvm::Expected<SelectedInstructionEntry>
selectInstructionEntry(const deployment::FinalizedDeployment &deployment,
                       dataflow::RootThreadLaunchRef root,
                       fabric::AccCoreOccurrenceRef accCore,
                       const ArtifactIdentity &fabricIdentity,
                       const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::optional<SelectedInstructionEntry> selected;
  for (const auto indexed :
       llvm::enumerate(deployment.deployment().instructionCoreBinaries())) {
    auto binary =
        importInstructionCoreBinary(indexed.value(), artifacts, blobs);
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

std::string spatialLaunchPath(std::size_t ordinal) {
  return "inputs/spatial-launch-" + std::to_string(ordinal) + ".bin";
}

std::string spatialChannelProjectionPath(std::size_t ordinal) {
  return "inputs/spatial-channel-" + std::to_string(ordinal) + ".bin";
}

std::string spatialChannelEnginePlanPath(std::size_t ordinal) {
  return "inputs/spatial-channel-engine-" + std::to_string(ordinal) + ".plan";
}

std::string spatialChannelBufferPath(std::size_t ordinal) {
  return "inputs/spatial-channel-buffer-" + std::to_string(ordinal) + ".init";
}

std::string spatialBridgeSocketPath(std::size_t ordinal) {
  return "outputs/spatial-bridge-" + std::to_string(ordinal) + ".sock";
}

std::string spatialBridgeResultPath(std::size_t ordinal) {
  return "outputs/spatial-bridge-" + std::to_string(ordinal) + ".result";
}

std::string mappedRtlLaunchPrefix(std::size_t ordinal) {
  return "rtl/launch-" + std::to_string(ordinal) + "/";
}

std::string mappedRtlLaunchResultPath(std::size_t ordinal) {
  constexpr llvm::StringLiteral outputRoot = "outputs/";
  assert(eda::open_source::mappedRtlResultPath.starts_with(outputRoot));
  return outputRoot.str() + mappedRtlLaunchPrefix(ordinal) +
         eda::open_source::mappedRtlResultPath.drop_front(outputRoot.size())
             .str();
}

llvm::Expected<std::optional<std::vector<std::vector<std::uint64_t>>>>
enumerateDenseCoordinates(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef graph) {
  auto logical = dataflow.projectWholeRootedGraphLogicalDomain(graph);
  if (!logical)
    return logical.takeError();
  if (!*logical ||
      (*logical)->kind != dataflow::ThreadDomainKind::DenseRectangular)
    return std::optional<std::vector<std::vector<std::uint64_t>>>{};
  auto root = dataflow.resolve(graph.rootThreadLaunch);
  if (!root)
    return root.takeError();
  auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(root->op);
  if (!launch)
    return invalid("root thread reference does not resolve a launch");
  std::vector<std::uint64_t> extents;
  extents.reserve(launch.getGridUpperBounds().size());
  std::uint64_t count = 1;
  for (mlir::Value bound : launch.getGridUpperBounds()) {
    mlir::Attribute constant;
    if (!mlir::matchPattern(bound, mlir::m_Constant(&constant)))
      return std::optional<std::vector<std::vector<std::uint64_t>>>{};
    auto integer = llvm::dyn_cast<mlir::IntegerAttr>(constant);
    if (!integer || integer.getValue().isNegative() ||
        integer.getValue().getActiveBits() > 64)
      return std::optional<std::vector<std::vector<std::uint64_t>>>{};
    const std::uint64_t extent = integer.getValue().getZExtValue();
    if (extent == 0)
      return std::vector<std::vector<std::uint64_t>>{};
    if (count > kMaximumDenseSpatialLaunches / extent)
      return std::optional<std::vector<std::vector<std::uint64_t>>>{};
    count *= extent;
    extents.push_back(extent);
  }
  if (count > kMaximumDenseSpatialLaunches)
    return std::optional<std::vector<std::vector<std::uint64_t>>>{};
  std::vector<std::vector<std::uint64_t>> coordinates;
  coordinates.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t linear = 0; linear != count; ++linear) {
    std::uint64_t remainder = linear;
    std::vector<std::uint64_t> point(extents.size(), 0);
    for (std::size_t dimension = extents.size(); dimension != 0; --dimension) {
      point[dimension - 1] = remainder % extents[dimension - 1];
      remainder /= extents[dimension - 1];
    }
    coordinates.push_back(std::move(point));
  }
  return std::optional<std::vector<std::vector<std::uint64_t>>>(
      std::move(coordinates));
}

llvm::Expected<std::vector<std::uint64_t>>
evaluateSourceMap(mlir::AffineMap map,
                  llvm::ArrayRef<std::uint64_t> consumerCoordinates) {
  if (!map || map.getNumDims() != consumerCoordinates.size() ||
      map.getNumSymbols() != 0)
    return invalid("channel source map has a foreign logical signature");
  // A rank-zero dense domain has one point, represented by the empty
  // coordinate tuple. MLIR's generic constantFold reports failure for maps
  // with no results, so preserve that valid affine-map case explicitly.
  if (map.getNumResults() == 0)
    return std::vector<std::uint64_t>{};
  llvm::SmallVector<mlir::Attribute> operands;
  operands.reserve(consumerCoordinates.size());
  for (std::uint64_t coordinate : consumerCoordinates) {
    if (coordinate >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      return invalid("channel coordinate exceeds the affine integer domain");
    operands.push_back(
        mlir::IntegerAttr::get(mlir::IndexType::get(map.getContext()),
                               static_cast<std::int64_t>(coordinate)));
  }
  llvm::SmallVector<mlir::Attribute> results;
  if (mlir::failed(map.constantFold(operands, results)) ||
      results.size() != map.getNumResults())
    return invalid("channel source map did not fold at a concrete point");
  std::vector<std::uint64_t> coordinates;
  coordinates.reserve(results.size());
  for (mlir::Attribute result : results) {
    auto integer = llvm::dyn_cast<mlir::IntegerAttr>(result);
    if (!integer || integer.getValue().isNegative() ||
        integer.getValue().getActiveBits() > 64)
      return invalid("channel source map produced an invalid coordinate");
    coordinates.push_back(integer.getValue().getZExtValue());
  }
  return coordinates;
}

const mapping::SystemServiceRealizationView *
findServiceRealization(const mapping::SystemMappingView &mapping,
                       const mapping::SystemServiceObligationKey &key) {
  const auto found =
      llvm::find_if(mapping.serviceRealizations(),
                    [&](const mapping::SystemServiceRealizationView &service) {
                      return service.key == key;
                    });
  return found == mapping.serviceRealizations().end() ? nullptr : &*found;
}

const mapping::SystemServicePlanView *
findServicePlan(const mapping::SystemServiceRealizationView &realization,
                std::uint64_t ordinal) {
  const auto found = llvm::find_if(
      realization.plans, [&](const mapping::SystemServicePlanView &plan) {
        return plan.ordinal == ordinal;
      });
  return found == realization.plans.end() ? nullptr : &*found;
}

llvm::Expected<std::shared_ptr<const sim::ImportedSystemSimulationInputs>>
importCachedSystemInputs(const ArtifactRootReference &workload,
                         const ArtifactRootReference &runtimeInput,
                         const ArtifactStore &artifacts,
                         const BlobStore &blobs) {
  const std::array<ArtifactRootReference, 2> references{workload, runtimeInput};
  return evaluation::importCachedArtifact<sim::ImportedSystemSimulationInputs>(
      artifacts, &blobs, references, [&] {
        return sim::importSystemSimulationInputs(workload, runtimeInput,
                                                 artifacts, blobs);
      });
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
  auto systemInputs = importCachedSystemInputs(
      *request.workload(), *request.runtimeInput(), artifacts, blobs);
  if (!systemInputs)
    return systemInputs.takeError();
  const sim::ImportedSystemSimulationInputs &inputs = **systemInputs;
  if (inputs.deployment.reference() != subjects->first)
    return invalid("System workload names a foreign Deployment");
  auto binding = importGem5SimulationBinding(subjects->second, artifacts);
  if (!binding)
    return binding.takeError();
  auto fabricRoot =
      fabric::importEntireFabricRoot(binding->binding().fabric(), artifacts);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto system = fabric::requireSystemRoot(fabricRoot->view());
  if (!system)
    return system.takeError();
  if (binding->binding().fabric().artifact !=
      inputs.deployment.deployment().systemMapping().artifact) {
    auto mapping = mapping::importSystemMapping(
        inputs.deployment.deployment().systemMapping(), artifacts);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().fabricIdentity() !=
        binding->binding().fabric().artifact)
      return invalid("gem5 binding and Deployment name different Fabric roots");
  }
  if (!supportsSystemMemorySurface(inputs))
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  const deployment::Deployment &deployment = inputs.deployment.deployment();
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
  if (contexts->spatialDomains.empty())
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  std::vector<PendingSpatialLaunch> pendingLaunches;
  std::vector<dataflow::RootedGraphLaunchRef> graphs;
  for (const mapping::SystemSpatialContextDomain &domain :
       contexts->spatialDomains)
    if (!llvm::is_contained(graphs, domain.graph))
      graphs.push_back(domain.graph);
  llvm::sort(graphs, [](const auto &lhs, const auto &rhs) {
    return std::tuple(lhs.rootThreadLaunch.entity.value(),
                      lhs.staticGraphLaunch.entity.value()) <
           std::tuple(rhs.rootThreadLaunch.entity.value(),
                      rhs.staticGraphLaunch.entity.value());
  });
  std::set<std::vector<std::uint8_t>> selectedAccCores;
  for (const dataflow::RootedGraphLaunchRef &graph : graphs) {
    auto coordinates = enumerateDenseCoordinates(*dataflowView, graph);
    if (!coordinates)
      return coordinates.takeError();
    if (!*coordinates || (*coordinates)->empty())
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    auto launch = dataflowView->resolve(graph.staticGraphLaunch);
    if (!launch)
      return launch.takeError();
    auto launchOp = llvm::dyn_cast<dataflow::GraphLaunchOp>(launch->op);
    if (!launchOp || !launchOp.getValueInputs().empty() ||
        !launchOp.getMemoryInputs().empty())
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    for (const std::vector<std::uint64_t> &point : **coordinates) {
      auto selection = deployment::resolveDeploymentSpatialLaunchSelection(
          inputs.deployment, graph, point, artifacts, blobs);
      if (!selection)
        return selection.takeError();
      const std::vector<std::uint8_t> coreKey = fabric::canonicalFabricBytes(
          fabric::SpatialCoreOccurrenceRef{selection->context.accCore});
      if (!selectedAccCores.insert(coreKey).second)
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      auto spatialMapping =
          mapping::importSpatialMapping(selection->spatialMapping, artifacts);
      if (!spatialMapping)
        return spatialMapping.takeError();
      std::vector<mapping::SystemPresburgerCell> contextDomain;
      for (const mapping::SystemSpatialContextDomain &domain :
           contexts->spatialDomains)
        if (domain.graph == graph && domain.context == selection->context)
          contextDomain.insert(contextDomain.end(), domain.cells.begin(),
                               domain.cells.end());
      if (contextDomain.empty())
        return invalid("selected Spatial context has no logical domain");
      pendingLaunches.push_back(
          PendingSpatialLaunch{graph,
                               point,
                               selection->context.accCore,
                               selection->context,
                               std::move(contextDomain),
                               {fabric::fabricArtifactSchema.identity.str(),
                                fabric::fabricArtifactSchema.version,
                                spatialMapping->view().fabricIdentity()},
                               selection->spatialMapping,
                               selection->hardwareImplementation,
                               {},
                               {},
                               {},
                               {},
                               {},
                               {},
                               {}});
    }
  }

  auto obligations = mapping::projectSystemServiceObligations(
      *dataflowView,
      systemMapping->view().executionBindings().rootThreadLaunches());
  if (!obligations)
    return obligations.takeError();
  std::vector<PendingChannelBuffer> channelBuffers;
  for (const mapping::SystemServiceObligationProjection &obligation :
       *obligations) {
    const auto *transfer =
        std::get_if<mapping::TransferObligationFamilyKey>(&obligation.key);
    if (!transfer)
      continue;
    const auto *channel =
        std::get_if<dataflow::ChannelProducerTerminalRef>(transfer);
    if (!channel)
      continue;
    const auto *producer =
        std::get_if<dataflow::GraphStreamOutputProducerRef>(&channel->producer);
    if (!producer)
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    const mapping::SystemServiceRealizationView *realization =
        findServiceRealization(systemMapping->view(), obligation.key);
    if (!realization || obligation.members.size() != 1 ||
        obligation.legs.size() != 1)
      return invalid("channel obligation has no unique service realization");
    const mapping::ServicePlanSelectionAnchor anchor =
        mapping::ServiceMemberPlanSelectionAnchor{obligation.members.front()};
    auto consumerBindings = dataflowView->channelConsumers(channel->producer);
    if (!consumerBindings)
      return consumerBindings.takeError();

    for (const auto producerIndexed : llvm::enumerate(pendingLaunches)) {
      PendingSpatialLaunch &producerLaunch = producerIndexed.value();
      if (producerLaunch.graph != producer->launch)
        continue;
      auto planOrdinal = mapping::selectSystemServicePlanOrdinal(
          *realization, anchor,
          mapping::ExecutionContextKey(producerLaunch.context),
          producerLaunch.contextDomain, producerLaunch.denseCoordinates);
      if (!planOrdinal)
        return planOrdinal.takeError();
      const mapping::SystemServicePlanView *plan =
          findServicePlan(*realization, *planOrdinal);
      if (!plan)
        return invalid("selected channel service plan is absent");

      struct DynamicSink final {
        std::size_t consumerLaunch = 0;
        std::uint64_t sinkOrdinal = 0;
        std::uint64_t streamInputOrdinal = 0;
      };
      std::vector<DynamicSink> dynamicSinks;
      for (const auto sinkIndexed : llvm::enumerate(obligation.sinks)) {
        const auto *sinkTerminal =
            std::get_if<dataflow::ChannelConsumerTerminalRef>(
                &sinkIndexed.value());
        if (!sinkTerminal)
          return invalid("channel obligation contains a non-channel sink");
        const auto *graphInput =
            std::get_if<dataflow::GraphStreamInputConsumerRef>(
                &sinkTerminal->consumer);
        if (!graphInput)
          return Gem5SystemFactsOrUnsupported{
              UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
        const auto binding = llvm::find_if(
            *consumerBindings,
            [&](const dataflow::ChannelConsumerBinding &candidate) {
              return candidate.consumer == sinkTerminal->consumer;
            });
        if (binding == consumerBindings->end() || !binding->sourceMap)
          return invalid("graph channel consumer has no source map");
        for (const auto consumerIndexed : llvm::enumerate(pendingLaunches)) {
          const PendingSpatialLaunch &consumerLaunch = consumerIndexed.value();
          if (consumerLaunch.graph != graphInput->launch)
            continue;
          auto source = evaluateSourceMap(*binding->sourceMap,
                                          consumerLaunch.denseCoordinates);
          if (!source)
            return source.takeError();
          if (*source != producerLaunch.denseCoordinates)
            continue;
          dynamicSinks.push_back({consumerIndexed.index(), sinkIndexed.index(),
                                  graphInput->ordinal});
        }
      }

      const auto leg =
          llvm::find_if(plan->transferLegs,
                        [&](const mapping::SystemTransferLegView &candidate) {
                          return candidate.leg == obligation.legs.front();
                        });
      if (dynamicSinks.empty()) {
        if (leg != plan->transferLegs.end() && !leg->sinks.empty())
          return invalid("empty channel instance selects a nonempty route");
        continue;
      }
      if (leg == plan->transferLegs.end())
        return invalid("live channel instance selects no transfer route");
      for (const DynamicSink &sink : dynamicSinks) {
        const mapping::SystemTransferTerminalKey expected =
            mapping::SystemTransferSinkTerminalKey{obligation.legs.front(),
                                                   sink.sinkOrdinal};
        if (!llvm::any_of(leg->sinks, [&](const auto &candidate) {
              return candidate.terminal == expected;
            }))
          return invalid("selected channel route omits a dynamic sink");
      }

      const std::size_t bufferOrdinal = channelBuffers.size();
      channelBuffers.push_back({producerIndexed.index(), producer->ordinal, 0});
      producerLaunch.channelOutputs.push_back(
          {producer->ordinal, bufferOrdinal});
      for (const DynamicSink &sink : dynamicSinks) {
        PendingSpatialLaunch &consumerLaunch =
            pendingLaunches[sink.consumerLaunch];
        if (llvm::any_of(consumerLaunch.channelInputs,
                         [&](const PendingChannelInput &input) {
                           return input.consumerStreamInputOrdinal ==
                                  sink.streamInputOrdinal;
                         }))
          return invalid("consumer stream input has multiple dynamic sources");
        consumerLaunch.channelInputs.push_back(
            {producerIndexed.index(), producer->ordinal,
             sink.streamInputOrdinal, bufferOrdinal});
      }
    }
  }

  for (PendingSpatialLaunch &pending : pendingLaunches) {
    auto launch = dataflowView->resolve(pending.graph.staticGraphLaunch);
    if (!launch)
      return launch.takeError();
    auto launchOp = llvm::cast<dataflow::GraphLaunchOp>(launch->op);
    if (pending.channelInputs.size() != launchOp.getStreamInputs().size())
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    sim::SpatialSimulationWorkload workloadDraft{pending.graph};
    workloadDraft.denseCoordinates = pending.denseCoordinates;
    for (const PendingChannelOutput &output : pending.channelOutputs)
      workloadDraft.observableContract.streamOutputs.push_back(
          output.producerStreamOutputOrdinal);
    llvm::sort(workloadDraft.observableContract.streamOutputs);
    auto workload =
        sim::finalizeSimulationWorkload(workloadDraft, *dataflowView);
    if (!workload)
      return workload.takeError();
    auto workloadReference =
        sim::publishSimulationWorkload(*workload, artifacts);
    if (!workloadReference)
      return workloadReference.takeError();
    sim::SpatialSimulationRuntimeInputDraft runtimeDraft{workload->identity()};
    runtimeDraft.runtimeStreams.resize(launchOp.getStreamInputs().size());
    auto runtime = sim::finalizeSimulationRuntimeInput(runtimeDraft, *workload,
                                                       *dataflowView);
    if (!runtime)
      return runtime.takeError();
    auto runtimeReference =
        sim::publishSimulationRuntimeInput(*runtime, artifacts);
    if (!runtimeReference)
      return runtimeReference.takeError();
    pending.spatialWorkload = std::move(*workloadReference);
    pending.spatialRuntimeInput = std::move(*runtimeReference);
    auto shapes = sim::projectSpatialSimulationBoundaryShapes(*dataflowView,
                                                              pending.graph);
    if (!shapes)
      return shapes.takeError();
    const auto projectWidths =
        [](llvm::ArrayRef<sim::SpatialSimulationValueShape> shapes)
        -> llvm::Expected<std::vector<std::uint32_t>> {
      std::vector<std::uint32_t> widths;
      widths.reserve(shapes.size());
      for (const sim::SpatialSimulationValueShape shape : shapes) {
        if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
            shape.lanesPerToken >
                std::numeric_limits<std::uint32_t>::max() / shape.laneBitWidth)
          return invalid("Spatial stream shape exceeds the channel wire");
        widths.push_back(static_cast<std::uint32_t>(shape.lanesPerToken) *
                         shape.laneBitWidth);
      }
      return widths;
    };
    auto inputWidths = projectWidths(shapes->streamInputs);
    if (!inputWidths)
      return inputWidths.takeError();
    auto outputWidths = projectWidths(shapes->streamOutputs);
    if (!outputWidths)
      return outputWidths.takeError();
    pending.streamInputBitWidths = std::move(*inputWidths);
    pending.streamOutputBitWidths = std::move(*outputWidths);
    pending.observableStreamOutputOrdinals =
        workloadDraft.observableContract.streamOutputs;
  }
  std::vector<Gem5ProcessorProjection> processors;
  std::vector<
      std::pair<fabric::AccCoreOccurrenceRef, Gem5SpatialBridgeParameters>>
      bridges;
  std::optional<Gem5SimpleMemoryParameters> memory;
  std::set<std::vector<std::uint8_t>> seenProcessors;
  std::set<std::vector<std::uint8_t>> seenMemories;
  for (const Gem5Correspondence &row : binding->binding().correspondences()) {
    if (const auto *processor =
            std::get_if<Gem5ProcessorCorrespondence>(&row)) {
      std::optional<Gem5ProcessorModelKind> model;
      if (processor->simObject.contract ==
          gem5ModelContractDescriptorRef(gem5RiscvTimingCpuModel()))
        model = Gem5ProcessorModelKind::TimingSimple;
      else if (processor->simObject.contract ==
               gem5ModelContractDescriptorRef(gem5RiscvO3CpuModel()))
        model = Gem5ProcessorModelKind::O3;
      if (!model)
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      if (!seenProcessors.insert(processor->simObject.payload).second)
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      auto parameters =
          decodeGem5RiscvCpuParameters(processor->simObject.payload);
      if (!parameters)
        return parameters.takeError();
      const fabric::InstructionCoreMicroarchitecturalRealization
          *microarchitecture = nullptr;
      std::visit(
          [&](const auto &core) {
            microarchitecture = system->instructionCoreMicroarchitecture(core);
          },
          processor->processor);
      if (!microarchitecture)
        return invalid("gem5 processor has no Fabric realization");
      const auto *outOfOrder = microarchitecture->outOfOrder();
      if ((*model == Gem5ProcessorModelKind::O3) != (outOfOrder != nullptr))
        return invalid("gem5 processor model disagrees with Fabric");
      processors.push_back(
          {processor->processor, *model, *parameters,
           microarchitecture->hardwareThreadCount(),
           std::vector<fabric::ExecutionUnitRecord>(
               microarchitecture->executionUnits().begin(),
               microarchitecture->executionUnits().end()),
           outOfOrder ? std::optional(*outOfOrder) : std::nullopt});
      continue;
    }
    if (const auto *spatial =
            std::get_if<Gem5SpatialBridgeCorrespondence>(&row)) {
      if (spatial->bridgeEndpoint.object.contract !=
          gem5ModelContractDescriptorRef(gem5SpatialBridgeModel()))
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      auto parameters = decodeGem5SpatialBridgeParameters(
          spatial->bridgeEndpoint.object.payload);
      if (!parameters)
        return parameters.takeError();
      const auto existing = llvm::find_if(bridges, [&](const auto &candidate) {
        return candidate.first == spatial->spatialCore.core;
      });
      if (existing == bridges.end())
        bridges.push_back({spatial->spatialCore.core, *parameters});
      else if (!(existing->second == *parameters))
        return Gem5SystemFactsOrUnsupported{
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
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
  if (processors.empty() || bridges.empty() || !memory)
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
  for (const Gem5ProcessorProjection &processor : processors) {
    if (std::holds_alternative<fabric::HostCoreOccurrenceRef>(
            processor.processor)) {
      if (hostProcessor)
        return invalid("gem5 binding contains more than one HostCore CPU");
      hostProcessor = &processor;
    }
  }
  if (!hostProcessor)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  const std::uint64_t hostCpuId = hostProcessor->parameters.cpuId;
  const auto *systemWorkload = inputs.workload.system();
  if (!systemWorkload)
    return invalid("imported System workload lost its typed payload");
  auto hostEntry = deployment::resolveDeploymentProgramEntry(
      inputs.deployment, systemWorkload->programEntryRef);
  if (!hostEntry)
    return hostEntry.takeError();

  auto memoryEndValue =
      checkedAdd(memory->baseAddress, memory->sizeBytes, "gem5 memory");
  if (!memoryEndValue)
    return memoryEndValue.takeError();
  const std::uint64_t memoryEnd = *memoryEndValue;
  std::uint64_t maximumBridgeEnd = 0;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> bridgeRanges;
  bridgeRanges.reserve(bridges.size());
  for (const auto &[core, bridge] : bridges) {
    (void)core;
    auto bridgeEndValue =
        checkedAdd(bridge.pioAddress, bridge.pioSize, "Spatial Bridge");
    if (!bridgeEndValue)
      return bridgeEndValue.takeError();
    if (memory->baseAddress < *bridgeEndValue && bridge.pioAddress < memoryEnd)
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    bridgeRanges.push_back({bridge.pioAddress, *bridgeEndValue});
    maximumBridgeEnd = std::max(maximumBridgeEnd, *bridgeEndValue);
  }
  llvm::sort(bridgeRanges);
  for (std::size_t ordinal = 1; ordinal != bridgeRanges.size(); ++ordinal)
    if (bridgeRanges[ordinal].first < bridgeRanges[ordinal - 1].second)
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  auto dispatchAddressValue =
      alignUp(maximumBridgeEnd, kGem5PageBytes, "Thread Dispatch");
  if (!dispatchAddressValue)
    return dispatchAddressValue.takeError();
  const std::uint64_t dispatchAddress = *dispatchAddressValue;
  auto dispatchEndValue = checkedAdd(
      dispatchAddress, kThreadDispatchApertureBytes, "Thread Dispatch");
  if (!dispatchEndValue)
    return dispatchEndValue.takeError();
  if (memory->baseAddress < *dispatchEndValue && dispatchAddress < memoryEnd)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  auto semanticInputs =
      materializeDeploymentPackage(inputs.deployment, artifacts, blobs);
  if (!semanticInputs)
    return semanticInputs.takeError();
  for (const PendingSpatialLaunch &launch : pendingLaunches) {
    if (!launch.spatialWorkload || !launch.spatialRuntimeInput)
      return invalid("pending Spatial launch has no finalized inputs");
    if (llvm::Error error = appendStoredObject(
            *semanticInputs, *launch.spatialWorkload, artifacts))
      return std::move(error);
    if (llvm::Error error = appendStoredObject(
            *semanticInputs, *launch.spatialRuntimeInput, artifacts))
      return std::move(error);
  }
  auto hostElf = blobs.get(deployment.hostProgram().programBlob());
  if (!hostElf)
    return hostElf.takeError();
  semanticInputs->push_back({kHostElfPath.str(), bytesToString(*hostElf),
                             inputs.deployment.reference(), true});
  std::vector<Gem5InstructionImage> instructionImages;
  instructionImages.reserve(deployment.instructionCoreBinaries().size());
  for (const auto indexed :
       llvm::enumerate(deployment.instructionCoreBinaries())) {
    auto binary =
        importInstructionCoreBinary(indexed.value(), artifacts, blobs);
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
  const auto &threadBytes =
      deployment.threadDispatchImage().canonicalBytes().bytes();
  const auto &admissionBytes =
      deployment.admissionImage().canonicalBytes().bytes();
  for (std::size_t ordinal = 0; ordinal != pendingLaunches.size(); ++ordinal)
    semanticInputs->push_back({spatialLaunchPath(ordinal),
                               bytesToString(launchBytes),
                               inputs.deployment.reference(), false});
  semanticInputs->push_back({kThreadDispatchPath.str(),
                             bytesToString(threadBytes),
                             inputs.deployment.reference(), false});
  semanticInputs->push_back({kAdmissionPath.str(),
                             bytesToString(admissionBytes),
                             inputs.deployment.reference(), false});

  auto midpointValue = checkedAdd(memory->baseAddress, memory->sizeBytes / 2,
                                  "gem5 runtime image arena");
  if (!midpointValue)
    return midpointValue.takeError();
  auto cursorValue =
      alignUp(*midpointValue, kGem5PageBytes, "gem5 runtime image arena");
  if (!cursorValue)
    return cursorValue.takeError();
  std::uint64_t cursor = *cursorValue;
  std::vector<Gem5RuntimeImage> runtimeImages;
  auto reserveRuntimeRange =
      [&](std::uint64_t size,
          llvm::StringRef role) -> llvm::Expected<std::uint64_t> {
    if (size == 0)
      return invalid(role + " is empty");
    const std::uint64_t address = cursor;
    auto end = checkedAdd(cursor, size, role);
    if (!end)
      return end.takeError();
    auto aligned = alignUp(*end, kGem5PageBytes, role);
    if (!aligned)
      return aligned.takeError();
    cursor = *aligned;
    return address;
  };
  auto placeRuntimeImage =
      [&](llvm::StringRef path,
          std::uint64_t size) -> llvm::Expected<std::uint64_t> {
    auto address = reserveRuntimeRange(size, "gem5 runtime image");
    if (!address)
      return address.takeError();
    runtimeImages.push_back({path.str(), *address});
    return *address;
  };
  auto threadAddress =
      placeRuntimeImage(kThreadDispatchPath, threadBytes.size());
  auto admissionAddress =
      placeRuntimeImage(kAdmissionPath, admissionBytes.size());
  if (!threadAddress || !admissionAddress)
    return llvm::joinErrors(threadAddress ? llvm::Error::success()
                                          : threadAddress.takeError(),
                            admissionAddress ? llvm::Error::success()
                                             : admissionAddress.takeError());
  std::vector<std::uint64_t> launchAddresses;
  launchAddresses.reserve(pendingLaunches.size());
  for (std::size_t ordinal = 0; ordinal != pendingLaunches.size(); ++ordinal) {
    auto address =
        placeRuntimeImage(spatialLaunchPath(ordinal), launchBytes.size());
    if (!address)
      return address.takeError();
    launchAddresses.push_back(*address);
  }

  for (const auto indexed : llvm::enumerate(channelBuffers)) {
    auto address = reserveRuntimeRange(kSpatialChannelBufferBytes,
                                       "gem5 Spatial channel buffer");
    if (!address)
      return address.takeError();
    indexed.value().address = *address;
    const std::string path = spatialChannelBufferPath(indexed.index());
    runtimeImages.push_back({path, *address});
    semanticInputs->push_back(
        {path, std::string(gem5SpatialChannelBufferHeaderBytes, '\0'),
         inputs.deployment.reference(), false});
  }

  std::vector<Gem5SpatialLaunchProjection> spatialLaunches;
  spatialLaunches.reserve(pendingLaunches.size());
  for (const auto indexed : llvm::enumerate(pendingLaunches)) {
    const PendingSpatialLaunch &pending = indexed.value();
    const Gem5ProcessorProjection *instructionProcessor = nullptr;
    for (const Gem5ProcessorProjection &processor : processors) {
      const auto *context =
          std::get_if<fabric::InstructionCoreContextRef>(&processor.processor);
      if (!context || context->core != pending.accCore)
        continue;
      if (instructionProcessor)
        return invalid("gem5 binding repeats a selected InstructionCore");
      instructionProcessor = &processor;
    }
    const Gem5SpatialBridgeParameters *selectedBridge = nullptr;
    for (const auto &[core, bridge] : bridges) {
      if (core != pending.accCore)
        continue;
      if (selectedBridge)
        return invalid("gem5 binding repeats a selected Spatial Bridge");
      selectedBridge = &bridge;
    }
    if (!instructionProcessor || !selectedBridge)
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    auto instruction = selectInstructionEntry(
        inputs.deployment, pending.graph.rootThreadLaunch, pending.accCore,
        systemMapping->view().fabricIdentity(), artifacts, blobs);
    if (!instruction)
      return instruction.takeError();
    Gem5DispatchTarget dispatch{instructionProcessor->parameters.cpuId,
                                instruction->imageOrdinal,
                                "__loom_thread_entry_" +
                                    std::to_string(instruction->entryOrdinal),
                                selectedBridge->pioAddress,
                                launchAddresses[indexed.index()],
                                static_cast<std::uint64_t>(launchBytes.size())};
    Gem5SpatialChannelProjection channelProjection;
    Gem5SpatialChannelEnginePlan enginePlan;
    channelProjection.inputs.reserve(pending.channelInputs.size());
    enginePlan.inputs.reserve(pending.channelInputs.size());
    for (const PendingChannelInput &input : pending.channelInputs) {
      if (input.producerLaunch >= pendingLaunches.size() ||
          input.buffer >= channelBuffers.size())
        return invalid(
            "Spatial channel input names an absent launch or buffer");
      const PendingSpatialLaunch &producer =
          pendingLaunches[input.producerLaunch];
      const PendingChannelBuffer &buffer = channelBuffers[input.buffer];
      if (!producer.spatialWorkload || !producer.spatialRuntimeInput)
        return invalid("Spatial channel producer has no finalized inputs");
      if (input.consumerStreamInputOrdinal >=
              pending.streamInputBitWidths.size() ||
          input.producerStreamOutputOrdinal >=
              producer.streamOutputBitWidths.size())
        return invalid("Spatial channel input has an absent stream shape");
      const auto observation =
          llvm::find(producer.observableStreamOutputOrdinals,
                     input.producerStreamOutputOrdinal);
      if (observation == producer.observableStreamOutputOrdinals.end())
        return invalid("Spatial channel producer output is not observable");
      const std::uint32_t producerWidth =
          producer.streamOutputBitWidths[input.producerStreamOutputOrdinal];
      const std::uint32_t consumerWidth =
          pending.streamInputBitWidths[input.consumerStreamInputOrdinal];
      if (producerWidth != consumerWidth)
        return invalid("Spatial channel changes token bit width");
      channelProjection.inputs.push_back(
          {*producer.spatialWorkload, *producer.spatialRuntimeInput,
           input.producerStreamOutputOrdinal, input.consumerStreamInputOrdinal,
           buffer.address, kSpatialChannelBufferBytes});
      enginePlan.inputs.push_back(
          {input.consumerStreamInputOrdinal,
           static_cast<std::uint64_t>(std::distance(
               producer.observableStreamOutputOrdinals.begin(), observation)),
           producerWidth, buffer.address, kSpatialChannelBufferBytes});
    }
    channelProjection.outputs.reserve(pending.channelOutputs.size());
    for (const PendingChannelOutput &output : pending.channelOutputs) {
      if (output.buffer >= channelBuffers.size())
        return invalid("Spatial channel output names an absent buffer");
      const PendingChannelBuffer &buffer = channelBuffers[output.buffer];
      channelProjection.outputs.push_back({output.producerStreamOutputOrdinal,
                                           buffer.address,
                                           kSpatialChannelBufferBytes});
      enginePlan.outputs.push_back(
          {buffer.address, kSpatialChannelBufferBytes});
    }
    auto encodedProjection =
        encodeGem5SpatialChannelProjection(std::move(channelProjection));
    if (!encodedProjection)
      return encodedProjection.takeError();
    const std::string channelPath =
        spatialChannelProjectionPath(indexed.index());
    const std::string enginePlanPath =
        spatialChannelEnginePlanPath(indexed.index());
    semanticInputs->push_back({channelPath, bytesToString(*encodedProjection),
                               inputs.deployment.reference(), false});
    semanticInputs->push_back(
        {enginePlanPath,
         encodeGem5SpatialChannelEnginePlan(std::move(enginePlan)),
         inputs.deployment.reference(), false});
    spatialLaunches.push_back(
        {pending.fabric, pending.spatialMapping, pending.hardwareImplementation,
         *pending.spatialWorkload, *pending.spatialRuntimeInput,
         std::move(channelPath), std::move(enginePlanPath),
         std::vector<std::uint8_t>(launchBytes.begin(), launchBytes.end()),
         std::move(dispatch), *selectedBridge});
  }

  const sim::SystemSimulationRuntimeInput &systemRuntime =
      *inputs.runtimeInput.system();
  std::vector<std::uint64_t> memoryObjectAddresses;
  memoryObjectAddresses.reserve(systemRuntime.memoryObjects.size());
  for (const auto indexed : llvm::enumerate(systemRuntime.memoryObjects)) {
    const sim::RuntimeMemoryObject &object = indexed.value();
    if (!object.pointerValues.empty() ||
        llvm::any_of(object.initialBytes,
                     [](const sim::SemanticMemoryByte &byte) {
                       return byte.state != sim::SemanticState::Defined;
                     }))
      return Gem5SystemFactsOrUnsupported{
          UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    const std::string path = "inputs/system-memory-object-" +
                             std::to_string(indexed.index()) + ".bin";
    auto address = placeRuntimeImage(path, object.initialBytes.size());
    if (!address)
      return address.takeError();
    memoryObjectAddresses.push_back(*address);
    std::string contents;
    contents.reserve(object.initialBytes.size());
    for (const sim::SemanticMemoryByte &byte : object.initialBytes)
      contents.push_back(static_cast<char>(byte.value));
    semanticInputs->push_back(
        {path, std::move(contents), *request.runtimeInput(), false});
  }

  std::vector<std::uint8_t> memoryTable{'L', 'G', 'M', 'I'};
  appendGuestU32(memoryTable, 1);
  appendGuestU64(memoryTable, systemRuntime.memoryInterfaceBindings.size());
  for (const sim::SystemMemoryInterfaceBindingEntry &bindingEntry :
       systemRuntime.memoryInterfaceBindings) {
    if (bindingEntry.binding.objectOrdinal >= memoryObjectAddresses.size())
      return invalid("System memory binding names an absent object");
    const sim::RuntimeMemoryObject &object =
        systemRuntime.memoryObjects[bindingEntry.binding.objectOrdinal];
    if (bindingEntry.binding.byteOffset >= object.initialBytes.size())
      return invalid("System memory binding offset is out of range");
    auto interface = deployment::resolveDeploymentExternalInterface(
        inputs.deployment, bindingEntry.interfaceRef);
    if (!interface)
      return interface.takeError();
    if ((*interface)->kind != deployment::HostExternalInterfaceKind::Memory)
      return invalid("System memory table contains a non-memory interface");
    std::uint32_t permissions = 0;
    if ((*interface)->direction !=
        deployment::HostExternalInterfaceDirection::Output)
      permissions |= 1;
    if ((*interface)->direction !=
        deployment::HostExternalInterfaceDirection::Input)
      permissions |= 2;
    auto address =
        checkedAdd(memoryObjectAddresses[bindingEntry.binding.objectOrdinal],
                   bindingEntry.binding.byteOffset, "System memory interface");
    if (!address)
      return address.takeError();
    appendGuestU64(memoryTable,
                   bindingEntry.interfaceRef.externalInterfaceOrdinal);
    appendGuestU64(memoryTable, *address);
    appendGuestU64(memoryTable, object.initialBytes.size() -
                                    bindingEntry.binding.byteOffset);
    appendGuestU32(memoryTable, permissions);
    appendGuestU32(memoryTable, 0);
  }

  std::uint64_t memoryTableAddress = 0;
  if (!systemRuntime.memoryInterfaceBindings.empty()) {
    auto address = placeRuntimeImage(kMemoryTablePath, memoryTable.size());
    if (!address)
      return address.takeError();
    memoryTableAddress = *address;
    semanticInputs->push_back({kMemoryTablePath.str(),
                               bytesToString(memoryTable),
                               *request.runtimeInput(), false});
  }

  std::vector<Gem5MemoryObservationProjection> memoryObservations;
  memoryObservations.reserve(
      systemWorkload->observableContract.memories.size());
  for (const sim::SystemMemoryObservable &observable :
       systemWorkload->observableContract.memories) {
    const sim::SystemMemoryInterfaceBindingEntry *bindingEntry =
        findMemoryBinding(systemRuntime, observable.interfaceRef);
    if (!bindingEntry ||
        bindingEntry->binding.objectOrdinal >= memoryObjectAddresses.size())
      return invalid("System memory observable has no runtime binding");
    const sim::RuntimeMemoryObject &object =
        systemRuntime.memoryObjects[bindingEntry->binding.objectOrdinal];
    auto address = checkedAdd(
        memoryObjectAddresses[bindingEntry->binding.objectOrdinal],
        bindingEntry->binding.byteOffset, "System memory observation");
    if (!address)
      return address.takeError();
    memoryObservations.push_back(
        {bindingEntry->binding.objectOrdinal, bindingEntry->binding.byteOffset,
         *address,
         object.initialBytes.size() - bindingEntry->binding.byteOffset,
         observable.form});
  }

  const std::uint64_t stackBase = cursor;
  std::uint64_t maximumCpuId = 0;
  for (const Gem5ProcessorProjection &processor : processors)
    maximumCpuId = std::max(maximumCpuId, processor.parameters.cpuId);
  auto stackCount = checkedAdd(maximumCpuId, 1, "gem5 stack count");
  if (!stackCount ||
      *stackCount > std::numeric_limits<std::uint64_t>::max() / kGem5StackBytes)
    return invalid("gem5 stack arena size overflows uint64");
  auto stackEnd =
      checkedAdd(stackBase, *stackCount * kGem5StackBytes, "gem5 stack arena");
  if (!stackEnd)
    return stackEnd.takeError();
  if (*stackEnd > memoryEnd)
    return Gem5SystemFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  llvm::sort(*semanticInputs, [](const auto &lhs, const auto &rhs) {
    return lhs.relativePath < rhs.relativePath;
  });

  return Gem5SystemFactsOrUnsupported{Gem5SystemFacts{
      selectedEngine(request), inputs.deployment.reference(),
      binding->reference(), std::move(dataflowReference),
      std::move(spatialLaunches), std::move(*semanticInputs),
      std::move(processors), (*hostEntry)->abiSymbol, hostCpuId,
      std::move(instructionImages), std::move(runtimeImages),
      memoryTableAddress, systemRuntime.memoryInterfaceBindings.size(),
      std::move(memoryObservations), dispatchAddress, stackBase,
      kGem5StackBytes, *memory}};
}

} // namespace loom::runtime::gem5_system
