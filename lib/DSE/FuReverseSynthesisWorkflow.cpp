#include "DSE/FuReverseSynthesisWorkflow.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/FuReverseSynthesis.h"
#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "PnR/PnrConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr std::uint64_t synthesisNodeOrdinal = 0;
constexpr std::uint64_t spatialPnrNodeOrdinal = 1;
constexpr std::uint64_t jointSpatialPnrNodeOrdinal = 2;
constexpr std::uint64_t systemPnrNodeOrdinal = 3;
constexpr std::uint64_t portableRtlNodeOrdinal = 4;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fu_reverse_synthesis_workflow_invalid: " +
                                     message);
}

llvm::Error unsupportedReachability(const llvm::Twine &message) {
  return llvm::make_error<FuReverseSynthesisError>(
      FuReverseSynthesisFailure::UnsupportedGraphReachability, message.str());
}

llvm::Error verifyFullyRootReachableGraphDomain(
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (dataflow.rootThreadLaunches().empty())
    return unsupportedReachability(
        "bounded reverse synthesis requires a rooted graph domain");

  std::vector<::dataflow::RootedGraphLaunchRef> rootedLaunches;
  dataflow.forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef launch) {
        rootedLaunches.push_back(launch);
      });
  std::vector<::dataflow::GraphRef> reachableGraphs;
  for (const auto launch : rootedLaunches) {
    auto graph = dataflow.resolve(launch);
    if (!graph)
      return graph.takeError();
    if (!llvm::is_contained(reachableGraphs, *graph))
      reachableGraphs.push_back(*graph);
  }
  if (reachableGraphs.empty() ||
      reachableGraphs.size() != dataflow.graphs().size())
    return unsupportedReachability(
        "every bounded synthesis graph must be reachable from a root thread");
  return llvm::Error::success();
}

llvm::Expected<ArtifactRootReference>
requireOne(const CompletedDsePlanExecution &execution, PlanOutputRef output,
           llvm::StringRef name) {
  if (!execution.hasOutput(output))
    return invalid(name + " output is absent from the completed plan");
  const llvm::ArrayRef<ArtifactRootReference> artifacts =
      execution.resolve(output);
  if (artifacts.size() != 1)
    return invalid(name + " output is not unique");
  return artifacts.front();
}

const ::loom::mapping::FinalizedTechMapping *
findTechMapping(llvm::ArrayRef<::loom::mapping::FinalizedTechMapping> mappings,
                const ArtifactIdentity &identity) {
  for (const auto &mapping : mappings)
    if (mapping.view().identity() == identity)
      return &mapping;
  return nullptr;
}

const ::loom::mapping::FinalizedSpatialMapping *findSpatialMapping(
    llvm::ArrayRef<::loom::mapping::FinalizedSpatialMapping> mappings,
    const ArtifactRootReference &reference) {
  for (const auto &mapping : mappings)
    if (mapping.reference() == reference)
      return &mapping;
  return nullptr;
}

llvm::Error verifyGraphBindingTarget(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef launch,
    const ArtifactRootReference &target,
    llvm::ArrayRef<::loom::mapping::FinalizedSpatialMapping> spatialMappings,
    llvm::ArrayRef<::loom::mapping::FinalizedTechMapping> techMappings) {
  auto graph = dataflow.resolve(launch);
  if (!graph)
    return graph.takeError();
  const auto *spatial = findSpatialMapping(spatialMappings, target);
  if (!spatial)
    return invalid("SystemMapping selects a foreign SpatialMapping");
  const auto *tech =
      findTechMapping(techMappings, spatial->view().techMappingIdentity());
  if (!tech || !llvm::is_contained(tech->view().covers(), *graph))
    return invalid("SystemMapping graph binding selects a SpatialMapping "
                   "that does not cover the rooted graph");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FuReverseSynthesisCandidateWorkflow>
buildFuReverseSynthesisCandidateWorkflow(const ArtifactRootReference &dataflow,
                                         const ResolvedConfig &baseConfig,
                                         const ArtifactStore &store) {
  if (!baseConfig.dse.planNodes.empty())
    return invalid("base ResolvedConfig already owns a DSE invocation plan");
  if (llvm::Error error = registerFuReverseSynthesisCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerRootCompleteSpatialPnrCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerRootCompleteSystemPnrCandidateGenerator())
    return std::move(error);
  if (llvm::Error error = registerPortableSpatialCoreRtlCandidateGenerator())
    return std::move(error);

  auto importedDataflow = ::dataflow::importCanonicalDataflow(dataflow, store);
  if (!importedDataflow)
    return importedDataflow.takeError();
  auto dataflowView = importedDataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  if (llvm::Error error = verifyFullyRootReachableGraphDomain(*dataflowView))
    return std::move(error);
  std::vector<::dataflow::GraphRef> graphDomain;
  graphDomain.reserve(dataflowView->graphs().size());
  for (const ::dataflow::CanonicalGraphView &graph : dataflowView->graphs())
    graphDomain.push_back(graph.ref);
  if (llvm::Error error = verifyScalarIntegerAddSubFuSynthesisDomain(
          *dataflowView, graphDomain))
    return std::move(error);

  auto techConfig =
      ::loom::mapping::projectResolvedTechMappingConfigView(baseConfig);
  if (!techConfig)
    return techConfig.takeError();
  auto spatialConfig =
      ::loom::pnr::projectResolvedSpatialPnrConfigView(baseConfig);
  if (!spatialConfig)
    return spatialConfig.takeError();
  auto systemConfig =
      ::loom::pnr::projectResolvedSystemPnrConfigView(baseConfig);
  if (!systemConfig)
    return systemConfig.takeError();
  auto rtlConfig = resolvePortableSpatialCoreRtlConfig();
  if (!rtlConfig)
    return rtlConfig.takeError();

  const PlanOutputRef module{
      synthesisNodeOrdinal,
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::Module)};
  const PlanOutputRef techMappings{
      synthesisNodeOrdinal,
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::TechMapping)};
  const PlanOutputRef jointTechMapping{
      synthesisNodeOrdinal,
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::JointTechMapping)};
  const PlanOutputRef system{
      synthesisNodeOrdinal,
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::System)};
  const PlanOutputRef timing{
      synthesisNodeOrdinal,
      static_cast<std::uint32_t>(
          FuReverseSynthesisOutput::PhysicalTimingProfile)};
  const PlanOutputRef abi{
      synthesisNodeOrdinal,
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::ConfigurationAbi)};
  const PlanOutputRef spatialMappings{spatialPnrNodeOrdinal, 0};
  const PlanOutputRef jointSpatialMappings{jointSpatialPnrNodeOrdinal, 0};
  const PlanOutputRef systemMappings{systemPnrNodeOrdinal, 0};
  const PlanOutputRef portableRtl{portableRtlNodeOrdinal, 0};

  ResolvedConfig config = baseConfig;
  config.dse.planNodes = {
      GeneratePlanNodeDefinition{
          fuReverseSynthesisCandidateGeneratorDescriptor().reference(),
          {ExactPlanArtifacts{{dataflow}}},
          techConfig->canonicalViewBytes().vec(),
          techConfig->digest()},
      GeneratePlanNodeDefinition{
          rootCompleteSpatialPnrCandidateGeneratorDescriptor().reference(),
          {techMappings, module, timing},
          spatialConfig->canonicalViewBytes().vec(),
          spatialConfig->digest()},
      GeneratePlanNodeDefinition{
          rootCompleteSpatialPnrCandidateGeneratorDescriptor().reference(),
          {jointTechMapping, module, timing},
          spatialConfig->canonicalViewBytes().vec(),
          spatialConfig->digest()},
      GeneratePlanNodeDefinition{
          rootCompleteSystemPnrCandidateGeneratorDescriptor().reference(),
          {ExactPlanArtifacts{{dataflow}}, jointSpatialMappings, system, timing,
           ExactPlanArtifacts{}, ExactPlanArtifacts{}},
          systemConfig->canonicalViewBytes().vec(),
          systemConfig->digest()},
      GeneratePlanNodeDefinition{
          portableSpatialCoreRtlCandidateGeneratorDescriptor().reference(),
          {system, abi, ExactPlanArtifacts{}},
          rtlConfig->canonicalViewBytes().vec(),
          rtlConfig->digest()},
  };
  auto admitted = projectResolvedDseConfigView(config);
  if (!admitted)
    return admitted.takeError();

  return FuReverseSynthesisCandidateWorkflow(
      dataflow, std::move(config), module, techMappings, jointTechMapping,
      system, timing, abi, spatialMappings, jointSpatialMappings,
      systemMappings, portableRtl);
}

llvm::Expected<FuReverseSynthesisWorkflowArtifacts>
projectFuReverseSynthesisWorkflowArtifacts(
    const FuReverseSynthesisCandidateWorkflow &workflow,
    const CompletedDsePlanExecution &execution, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  auto disposition = classifyFuReverseSynthesisWorkflow(workflow, execution);
  if (!disposition)
    return disposition.takeError();
  if (*disposition != FuReverseSynthesisWorkflowDisposition::CompleteCandidate)
    return invalid("completed execution has no complete workflow candidate");

  auto module = requireOne(execution, workflow.module(), "Module");
  if (!module)
    return module.takeError();
  auto system = requireOne(execution, workflow.system(), "System");
  if (!system)
    return system.takeError();
  auto abi =
      requireOne(execution, workflow.configurationAbi(), "ConfigurationABI");
  if (!abi)
    return abi.takeError();
  auto jointTechMapping =
      requireOne(execution, workflow.jointTechMapping(), "joint TechMapping");
  if (!jointTechMapping)
    return jointTechMapping.takeError();

  FuReverseSynthesisWorkflowArtifacts projected{
      workflow.dataflow(),
      *module,
      execution.resolve(workflow.techMappings()).vec(),
      *jointTechMapping,
      *system,
      execution.resolve(workflow.physicalTimingProfiles()).vec(),
      *abi,
      execution.resolve(workflow.spatialMappings()).vec(),
      execution.resolve(workflow.jointSpatialMappings()).vec(),
      execution.resolve(workflow.systemMappings()).vec(),
      execution.resolve(workflow.portableRtlImplementations()).vec()};
  if (llvm::Error error = verifyFuReverseSynthesisWorkflowArtifacts(
          projected, artifacts, blobs))
    return std::move(error);
  return projected;
}

llvm::Expected<FuReverseSynthesisWorkflowDisposition>
classifyFuReverseSynthesisWorkflow(
    const FuReverseSynthesisCandidateWorkflow &workflow,
    const CompletedDsePlanExecution &execution) {
  auto view = projectResolvedDseConfigView(workflow.resolvedConfig());
  if (!view)
    return view.takeError();
  if (execution.resolvedDseConfigViewDigest() != view->digest())
    return invalid("completed execution belongs to another resolved plan");

  const std::array<PlanOutputRef, 10> required = {
      workflow.module(),
      workflow.techMappings(),
      workflow.jointTechMapping(),
      workflow.system(),
      workflow.physicalTimingProfiles(),
      workflow.configurationAbi(),
      workflow.spatialMappings(),
      workflow.jointSpatialMappings(),
      workflow.systemMappings(),
      workflow.portableRtlImplementations()};
  if (llvm::any_of(required, [&](PlanOutputRef output) {
        return !execution.hasOutput(output) ||
               execution.resolve(output).empty();
      }))
    return FuReverseSynthesisWorkflowDisposition::NoFeasibleCandidate;
  return FuReverseSynthesisWorkflowDisposition::CompleteCandidate;
}

llvm::Error verifyFuReverseSynthesisWorkflowArtifacts(
    const FuReverseSynthesisWorkflowArtifacts &artifacts,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto importedDataflow =
      ::dataflow::importCanonicalDataflow(artifacts.dataflow, store);
  if (!importedDataflow)
    return importedDataflow.takeError();
  auto dataflow = importedDataflow->view();
  if (!dataflow)
    return dataflow.takeError();
  if (llvm::Error error = verifyFullyRootReachableGraphDomain(*dataflow))
    return error;
  std::vector<::dataflow::GraphRef> synthesisGraphs;
  synthesisGraphs.reserve(dataflow->graphs().size());
  for (const ::dataflow::CanonicalGraphView &graph : dataflow->graphs())
    synthesisGraphs.push_back(graph.ref);
  auto module = ::loom::fabric::importEntireFabricRoot(artifacts.module, store);
  if (!module)
    return module.takeError();
  if (llvm::Error error = verifyScalarIntegerAddSubFuFabricLineage(
          *dataflow, synthesisGraphs, *module, store))
    return error;
  auto system = ::loom::fabric::importEntireFabricRoot(artifacts.system, store);
  if (!system)
    return system.takeError();
  if (llvm::Error error =
          verifyScalarIntegerAddSubFuSystemIdentity(*module, *system, store))
    return error;
  auto systemView = ::loom::fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();

  if (artifacts.physicalTimingProfiles.size() != 1)
    return invalid("workflow has no unique physical timing profile");
  if (llvm::Error error = verifyScalarIntegerAddSubFuPhysicalTimingLineage(
          *module, artifacts.physicalTimingProfiles.front(), store))
    return error;

  if (llvm::Error error = verifyScalarIntegerAddSubFuConfigurationAbiLineage(
          *system, artifacts.configurationAbi, store))
    return error;
  ::loom::hardware::ConfigurationABIImportSession abiSession(
      ::loom::hardware::ConfigurationABIImportSessionMode::Isolated);
  auto abi = ::loom::hardware::importConfigurationABI(
      artifacts.configurationAbi, store);
  if (!abi)
    return abi.takeError();

  std::vector<::loom::mapping::FinalizedTechMapping> techMappings;
  techMappings.reserve(artifacts.techMappings.size());
  std::vector<::dataflow::GraphRef> coveredGraphs;
  for (const auto &reference : artifacts.techMappings) {
    auto mapping = ::loom::mapping::importTechMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    if (llvm::Error error = verifyScalarIntegerAddSubFuMappingLineage(
            *dataflow, synthesisGraphs, *module, *mapping, store))
      return error;
    if (mapping->view().covers().size() != 1 ||
        llvm::is_contained(coveredGraphs, mapping->view().covers().front()))
      return invalid("workflow TechMapping graph coverage is not exact");
    coveredGraphs.push_back(mapping->view().covers().front());
    techMappings.push_back(std::move(*mapping));
  }
  if (coveredGraphs.size() != dataflow->graphs().size())
    return invalid("workflow TechMappings do not cover every graph");
  for (const auto &graph : dataflow->graphs())
    if (!llvm::is_contained(coveredGraphs, graph.ref))
      return invalid("workflow TechMappings omit an input graph");

  auto jointTechMapping =
      ::loom::mapping::importTechMapping(artifacts.jointTechMapping, store);
  if (!jointTechMapping)
    return jointTechMapping.takeError();
  if (llvm::Error error = verifyScalarIntegerAddSubFuJointMappingLineage(
          *dataflow, synthesisGraphs, *module, *jointTechMapping, store))
    return error;
  const llvm::ArrayRef<::loom::mapping::FinalizedTechMapping> jointTechMappings(
      &*jointTechMapping, 1);

  std::vector<::loom::mapping::FinalizedSpatialMapping> spatialMappings;
  spatialMappings.reserve(artifacts.spatialMappings.size());
  std::vector<ArtifactIdentity> spatializedTechMappings;
  for (const auto &reference : artifacts.spatialMappings) {
    auto mapping = ::loom::mapping::importSpatialMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflow->identity() ||
        mapping->view().fabricIdentity() != module->view().identity())
      return invalid("workflow SpatialMapping has foreign owners");
    if (!findTechMapping(techMappings, mapping->view().techMappingIdentity()))
      return invalid("workflow SpatialMapping selects a foreign TechMapping");
    if (!llvm::is_contained(spatializedTechMappings,
                            mapping->view().techMappingIdentity()))
      spatializedTechMappings.push_back(mapping->view().techMappingIdentity());
    spatialMappings.push_back(std::move(*mapping));
  }
  if (spatialMappings.empty() ||
      spatializedTechMappings.size() != techMappings.size())
    return invalid("workflow SpatialMappings do not establish every TechMap");

  std::vector<::loom::mapping::FinalizedSpatialMapping> jointSpatialMappings;
  jointSpatialMappings.reserve(artifacts.jointSpatialMappings.size());
  for (const auto &reference : artifacts.jointSpatialMappings) {
    auto mapping = ::loom::mapping::importSpatialMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflow->identity() ||
        mapping->view().fabricIdentity() != module->view().identity() ||
        mapping->view().techMappingIdentity() !=
            jointTechMapping->view().identity())
      return invalid("workflow joint SpatialMapping has foreign owners");
    jointSpatialMappings.push_back(std::move(*mapping));
  }
  if (jointSpatialMappings.empty())
    return invalid("workflow produced no joint SpatialMapping");

  if (artifacts.systemMappings.empty())
    return invalid("workflow produced no complete SystemMapping");
  ::loom::mapping::SystemMappingImportSession mappingSession(
      store, artifacts.systemMappings.size(),
      ::loom::mapping::SystemMappingImportSessionMode::New);
  std::vector<::dataflow::RootedGraphLaunchRef> rootedLaunches;
  dataflow->forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef launch) {
        rootedLaunches.push_back(launch);
      });
  std::vector<::dataflow::RootThreadLaunchRef> rootThreads;
  for (const auto &root : dataflow->rootThreadLaunches())
    rootThreads.push_back(root.ref);
  for (const auto &reference : artifacts.systemMappings) {
    auto mapping = ::loom::mapping::importSystemMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflow->identity() ||
        mapping->view().fabricIdentity() != system->view().identity())
      return invalid("workflow SystemMapping has foreign owners");
    const auto &execution = mapping->view().executionBindings();
    if (execution.rootThreadLaunches() !=
            llvm::ArrayRef<::dataflow::RootThreadLaunchRef>(rootThreads) ||
        execution.graphBindings().size() != rootedLaunches.size())
      return invalid("workflow SystemMapping does not cover the rooted domain");
    if (execution.spatialMappingImports().empty())
      return invalid("workflow SystemMapping imports no SpatialMapping");
    for (const auto &spatial : execution.spatialMappingImports())
      if (!findSpatialMapping(jointSpatialMappings, spatial))
        return invalid("workflow SystemMapping imports a foreign mapping");
    std::vector<::dataflow::RootedGraphLaunchRef> boundLaunches;
    boundLaunches.reserve(execution.graphBindings().size());
    for (const auto &binding : execution.graphBindings()) {
      if (!llvm::is_contained(rootedLaunches, binding.key))
        return invalid("workflow SystemMapping binds a foreign graph launch");
      if (llvm::is_contained(boundLaunches, binding.key))
        return invalid("workflow SystemMapping repeats a graph launch");
      boundLaunches.push_back(binding.key);
      std::size_t targetCount = 0;
      for (const auto &clause : binding.clauses) {
        if (llvm::Error error = verifyGraphBindingTarget(
                *dataflow, binding.key, clause.target, jointSpatialMappings,
                jointTechMappings))
          return error;
        ++targetCount;
      }
      if (binding.defaultTarget) {
        if (llvm::Error error = verifyGraphBindingTarget(
                *dataflow, binding.key, *binding.defaultTarget,
                jointSpatialMappings, jointTechMappings))
          return error;
        ++targetCount;
      }
      for (const auto &entry : binding.stableKeyEntries) {
        if (llvm::Error error = verifyGraphBindingTarget(
                *dataflow, binding.key, entry.target, jointSpatialMappings,
                jointTechMappings))
          return error;
        ++targetCount;
      }
      if (targetCount == 0)
        return invalid("workflow SystemMapping graph binding has no target");
    }
    if (boundLaunches.size() != rootedLaunches.size())
      return invalid("workflow SystemMapping omits a rooted graph launch");
    auto configured =
        ::loom::mapping::deriveConfiguredHardwareProjection(*mapping, store);
    if (!configured)
      return configured.takeError();
  }

  const auto accCores = systemView->artifact().accCoreOccurrences();
  if (artifacts.portableRtlImplementations.size() != accCores.size())
    return invalid("workflow portable RTL does not cover every SpatialCore");
  std::vector<::loom::fabric::SpatialCoreOccurrenceRef> rtlSubjects;
  for (const auto &reference : artifacts.portableRtlImplementations) {
    auto implementation =
        ::loom::hardware::importHardwareImplementation(reference, store, blobs);
    if (!implementation)
      return implementation.takeError();
    const auto &view = implementation->implementation();
    if (view.fabric() != artifacts.system ||
        view.configurationAbi() != artifacts.configurationAbi ||
        view.representationRoot().variant !=
            ::loom::hardware::RepresentationRootVariant::Rtl ||
        view.implementationPlatform())
      return invalid("workflow portable RTL has foreign implementation "
                     "lineage");
    if (llvm::is_contained(rtlSubjects, view.subject()))
      return invalid("workflow portable RTL repeats a SpatialCore subject");
    if (!llvm::any_of(accCores, [&](auto core) {
          return view.subject() ==
                 ::loom::fabric::SpatialCoreOccurrenceRef{core};
        }))
      return invalid("workflow portable RTL selects a foreign SpatialCore");
    if (llvm::Error error = ::loom::hardware::rtl::
            verifyPortableSpatialCoreHardwareImplementation(*abi,
                                                            *implementation))
      return error;
    rtlSubjects.push_back(view.subject());
  }
  return llvm::Error::success();
}

} // namespace loom::dse
