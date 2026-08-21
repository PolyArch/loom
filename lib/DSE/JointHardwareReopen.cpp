#include "DSE/JointHardwareReopen.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/ProductionOwners.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "DSE/TechMappingHardwareFeedback.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SpatialMappingHardwareDemand.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"
#include "PnR/PnrDerivedContext.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_hardware_reopen_invalid: " + message);
}

struct TechHardwareFeedbackObservation final {
  ArtifactRootReference module;
  mapping::TechMappingComputeContextHallDeficit feedback;
};

struct SpatialHardwareFeedbackObservation final {
  mapping::SpatialGraphBoundaryEndpointHallDeficit feedback;
};

struct SystemHardwareFeedbackObservation final {
  mapping::SystemAccCoreCapacityPressure feedback;
};

struct HardwareRecipeGrowth final {
  ResolvedConfig config;
  std::optional<ArtifactRootReference> accCoreParent;
  std::optional<ArtifactRootReference> accCoreTargetModule;
  std::uint64_t addedContexts = 0;
  std::uint64_t resultingContexts = 0;
  std::uint64_t addedGateways = 0;
  std::uint64_t resultingGateways = 0;
  std::uint64_t addedAccCores = 0;
  std::uint64_t resultingAccCores = 0;
};

struct MaterializedHardwareCandidate final {
  ArtifactRootReference reference;
  ResolvedConfig config;
  std::optional<pnr::SystemExecutionBindingCorrespondence>
      executionBindingCorrespondence;
  std::uint64_t addedContexts = 0;
  std::uint64_t resultingContexts = 0;
  std::uint64_t addedGateways = 0;
  std::uint64_t resultingGateways = 0;
  std::uint64_t addedAccCores = 0;
  std::uint64_t resultingAccCores = 0;
};

const dse::CompletedDsePlanExecution &
availableExecution(const dse::DsePlanExecutionResult &execution) {
  if (const auto *completed =
          std::get_if<dse::CompletedDsePlanExecution>(&execution))
    return *completed;
  return std::get<dse::IncompleteDsePlanExecution>(execution)
      .availableExecution();
}

std::size_t mappingCount(const dse::JointDesignExecution &execution) {
  std::size_t count = 0;
  for (const dse::JointMappedPair &pair : execution.mappedPairs)
    count += pair.systemMappings.size();
  return count;
}

void canonicalizeRoots(std::vector<ArtifactRootReference> &roots) {
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

llvm::Expected<std::vector<ArtifactRootReference>>
resolveSpatialMappingFrontier(const JointDesignExplorationPlan &plan,
                              const JointDesignExecution &execution) {
  if (plan.pairOutputs.size() != 1)
    return invalid("Spatial frontier reuse requires one exact Mapping pair");
  const CompletedDsePlanExecution &available =
      availableExecution(execution.planExecution);
  std::vector<ArtifactRootReference> mappings;
  for (const PlanOutputRef output : plan.pairOutputs.front().spatialMappings) {
    if (!available.hasOutput(output))
      return invalid("failed Mapping execution has no reusable Spatial "
                     "frontier");
    const auto roots = available.resolve(output);
    mappings.insert(mappings.end(), roots.begin(), roots.end());
  }
  canonicalizeRoots(mappings);
  if (mappings.empty())
    return invalid("failed Mapping execution has an empty Spatial frontier");
  return mappings;
}

llvm::Error bindImmutableSpatialMappingFrontier(
    JointDesignExplorationPlan &plan,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &artifacts) {
  constexpr std::size_t spatialMappingInputOrdinal = 1;
  if (plan.pairOutputs.size() != 1 || spatialMappings.empty())
    return invalid("Spatial frontier reuse requires one nonempty exact pair");
  JointDesignPlanPair &pair = plan.pairOutputs.front();
  if (pair.systemMappings.producerNodeOrdinal >=
      plan.resolvedConfig.dse.planNodes.size())
    return invalid("System Mapping output names a foreign plan node");
  auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &plan.resolvedConfig.dse
           .planNodes[pair.systemMappings.producerNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference() ||
      systemNode->inputBindings.size() <= spatialMappingInputOrdinal)
    return invalid("joint plan has no canonical application System provider");

  auto targetModules =
      projectJointDesignTargetModules(pair.pair.system, artifacts);
  if (!targetModules)
    return targetModules.takeError();
  std::vector<ArtifactRootReference> canonicalMappings(spatialMappings.begin(),
                                                       spatialMappings.end());
  const std::size_t originalCount = canonicalMappings.size();
  canonicalizeRoots(canonicalMappings);
  if (canonicalMappings.size() != originalCount)
    return invalid("reused Spatial frontier contains a duplicate root");
  const auto *join = std::get_if<BoundedPlanOutputJoin>(
      &systemNode->inputBindings[spatialMappingInputOrdinal]);
  if (!join || canonicalMappings.size() > join->maximumArtifacts)
    return invalid("reused Spatial frontier exceeds its exact plan bound");
  for (const ArtifactRootReference &reference : canonicalMappings) {
    auto mapping = mapping::importSpatialMapping(reference, artifacts);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() !=
        pair.pair.software.dataflow.artifact)
      return invalid("reused SpatialMapping has a foreign Dataflow owner");
    if (!llvm::any_of(*targetModules, [&](const auto &module) {
          return module.artifact == mapping->view().fabricIdentity();
        }))
      return invalid("reused SpatialMapping has a foreign Module owner");
  }

  GeneratePlanNodeDefinition retainedSystemNode = *systemNode;
  retainedSystemNode.inputBindings[spatialMappingInputOrdinal] =
      ExactPlanArtifacts{canonicalMappings};
  plan.resolvedConfig.dse.planNodes = {std::move(retainedSystemNode)};
  pair.techMappings.clear();
  pair.spatialMappings.clear();
  pair.systemMappings = PlanOutputRef{0, 0};
  auto admitted = projectResolvedDseConfigView(plan.resolvedConfig);
  if (!admitted)
    return admitted.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "reuse_immutable_spatial_frontier";
        fields["spatial_mapping_count"] = canonicalMappings.size();
      });
  return llvm::Error::success();
}

llvm::Error
bindSystemMappingMigrationSeed(JointDesignExplorationPlan &plan,
                               const ArtifactRootReference &migrationSeed,
                               const ArtifactStore &artifacts) {
  constexpr std::size_t migrationSeedInputOrdinal = 5;
  if (migrationSeed.schemaIdentity !=
          pnr::systemMappingCheckpointMigrationSeedArtifactSchema.identity ||
      migrationSeed.schemaVersion !=
          pnr::systemMappingCheckpointMigrationSeedArtifactSchema.version)
    return invalid("System migration seed has a foreign schema");
  auto imported =
      pnr::importSystemMappingCheckpointMigrationSeed(migrationSeed, artifacts);
  if (!imported)
    return imported.takeError();
  if (plan.pairOutputs.size() != 1)
    return invalid("System migration requires one exact Mapping pair");
  JointDesignPlanPair &pair = plan.pairOutputs.front();
  if (pair.systemMappings.producerNodeOrdinal >=
      plan.resolvedConfig.dse.planNodes.size())
    return invalid("System migration output names a foreign plan node");
  auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &plan.resolvedConfig.dse
           .planNodes[pair.systemMappings.producerNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference() ||
      systemNode->inputBindings.size() <= migrationSeedInputOrdinal)
    return invalid("joint plan has no canonical migration-seed input");
  systemNode->inputBindings[migrationSeedInputOrdinal] =
      ExactPlanArtifacts{{migrationSeed}};
  auto admitted = projectResolvedDseConfigView(plan.resolvedConfig);
  if (!admitted)
    return admitted.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "bind_system_mapping_migration_seed";
        fields["checkpoint"] = formatArtifactIdentityHex(
            imported->checkpoint().reference().artifact);
        fields["migration_seed"] =
            formatArtifactIdentityHex(migrationSeed.artifact);
        fields["preserved_acc_core_correspondences"] =
            imported->correspondence().accCores().size();
      });
  return llvm::Error::success();
}

llvm::Expected<dse::DsePlanExecutionResult> executeResolvedGeneratePlan(
    const ResolvedConfig &config,
    std::vector<ArtifactRootReference> semanticInputs,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  canonicalizeRoots(semanticInputs);
  auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                       canonicalResolvedConfigBytes(config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(config))
    return invalid("ResolvedConfig publication changed its identity");
  auto closure = dse::DseRunClosure::get(request.producer, semanticInputs,
                                         config, evidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto configView = dse::projectResolvedDseConfigView(config);
  if (!configView)
    return configView.takeError();
  llvm::SmallString<256> journalPath(request.journalRoot);
  llvm::sys::path::append(journalPath,
                          llvm::toHex(closure->runKey().bytes(), true));
  if (std::error_code error = llvm::sys::fs::create_directories(journalPath))
    return invalid("cannot create hardware-reopen journal: " + error.message());
  auto journal = dse::openExecutionJournal(journalPath, *closure, *configView);
  if (!journal)
    return journal.takeError();
  return dse::resumeDsePlan(*configView, *closure, *journal, scheduler,
                            request.executionPolicy, artifacts, blobs);
}

llvm::Expected<dse::JointDesignExecution>
executeJointPlan(const dse::JointDesignExplorationPlan &plan,
                 llvm::ArrayRef<ArtifactRootReference> evidence,
                 const JointHardwareReopenRequest &request,
                 dse::SiteScheduler &scheduler, const ArtifactStore &artifacts,
                 const BlobStore &blobs) {
  const ResolvedConfig &config = plan.resolvedConfig;
  auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                       canonicalResolvedConfigBytes(config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(config))
    return invalid("ResolvedConfig publication changed its identity");
  std::vector<ArtifactRootReference> semanticInputs =
      dse::projectJointDesignSemanticInputs(plan);
  auto closure = dse::DseRunClosure::get(request.producer, semanticInputs,
                                         config, evidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto configView = dse::projectResolvedDseConfigView(config);
  if (!configView)
    return configView.takeError();
  llvm::SmallString<256> journalPath(request.journalRoot);
  llvm::sys::path::append(journalPath,
                          llvm::toHex(closure->runKey().bytes(), true));
  if (std::error_code error = llvm::sys::fs::create_directories(journalPath))
    return invalid("cannot create Mapping alternative journal: " +
                   error.message());
  auto journal = dse::openExecutionJournal(journalPath, *closure, *configView);
  if (!journal)
    return journal.takeError();
  return dse::executeJointDesignExploration(plan, *closure, *journal, scheduler,
                                            request.executionPolicy, artifacts,
                                            blobs);
}

llvm::Expected<std::optional<TechHardwareFeedbackObservation>>
selectTechHardwareFeedback(const dse::JointDesignExecution &execution,
                           const ArtifactStore &artifacts) {
  const dse::CompletedDsePlanExecution &available =
      availableExecution(execution.planExecution);
  std::optional<TechHardwareFeedbackObservation> selected;
  for (const dse::GenerateInvocationFeedback &feedback :
       available.generateFeedback()) {
    const auto invocation = llvm::find_if(
        available.generateInvocations(), [&](const auto &candidate) {
          return candidate.planNodeOrdinal == feedback.planNodeOrdinal;
        });
    if (invocation == available.generateInvocations().end())
      return invalid("Generate feedback has no invocation owner");
    const dse::CandidateGeneratorDescriptor *descriptor =
        invocation->generatorBinding.descriptorRef().descriptor();
    if (!descriptor || !descriptor->ownerFeedbackPayload ||
        descriptor->ownerFeedbackPayload->schemaDescriptorBytes !=
            mapping::techMappingComputeContextHallFeedbackSchemaBytes())
      continue;

    std::optional<ArtifactRootReference> moduleReference;
    std::optional<fabric::FinalizedFabricRoot> module;
    for (const dse::CandidateGeneratorInputBinding &binding :
         invocation->inputBindings)
      for (const ArtifactRootReference &input : binding.artifacts) {
        if (input.schemaIdentity != fabric::fabricArtifactSchema.identity ||
            input.schemaVersion != fabric::fabricArtifactSchema.version)
          continue;
        auto imported = fabric::importEntireFabricRoot(input, artifacts);
        if (!imported)
          return imported.takeError();
        if (imported->view().rootKind() != fabric::FabricRootKind::Module)
          continue;
        if (moduleReference)
          return invalid("TechMapping feedback names multiple Module inputs");
        moduleReference = input;
        module = std::move(*imported);
      }
    if (!moduleReference || !module)
      return invalid("TechMapping feedback has no exact Module input");
    auto adopted = mapping::adoptTechMappingComputeContextHallFeedback(
        feedback.canonicalPayload, module->view());
    if (!adopted)
      return adopted.takeError();
    TechHardwareFeedbackObservation candidate{*moduleReference,
                                              std::move(*adopted)};
    if (!selected ||
        candidate.feedback.deficit() > selected->feedback.deficit() ||
        (candidate.feedback.deficit() == selected->feedback.deficit() &&
         artifactRootReferenceLess(candidate.module, selected->module)))
      selected = std::move(candidate);
  }
  return selected;
}

llvm::Expected<std::optional<SpatialHardwareFeedbackObservation>>
selectSpatialHardwareFeedback(const dse::JointDesignExecution &execution,
                              const ArtifactStore &artifacts) {
  const dse::CompletedDsePlanExecution &available =
      availableExecution(execution.planExecution);
  std::optional<SpatialHardwareFeedbackObservation> selected;
  for (const dse::GenerateInvocationFeedback &feedback :
       available.generateFeedback()) {
    const auto invocation = llvm::find_if(
        available.generateInvocations(), [&](const auto &candidate) {
          return candidate.planNodeOrdinal == feedback.planNodeOrdinal;
        });
    if (invocation == available.generateInvocations().end())
      return invalid("Generate feedback has no invocation owner");
    const dse::CandidateGeneratorDescriptor *descriptor =
        invocation->generatorBinding.descriptorRef().descriptor();
    if (!descriptor || !descriptor->ownerFeedbackPayload ||
        descriptor->ownerFeedbackPayload->schemaDescriptorBytes !=
            mapping::spatialGraphBoundaryEndpointHallFeedbackSchemaBytes())
      continue;

    std::optional<ArtifactRootReference> moduleReference;
    std::vector<ArtifactRootReference> techMappings;
    for (const dse::CandidateGeneratorInputBinding &binding :
         invocation->inputBindings)
      for (const ArtifactRootReference &input : binding.artifacts) {
        if (input.schemaIdentity == fabric::fabricArtifactSchema.identity &&
            input.schemaVersion == fabric::fabricArtifactSchema.version) {
          auto imported = fabric::importEntireFabricRoot(input, artifacts);
          if (!imported)
            return imported.takeError();
          if (imported->view().rootKind() != fabric::FabricRootKind::Module)
            continue;
          if (moduleReference)
            return invalid(
                "SpatialMapping feedback names multiple Module inputs");
          moduleReference = input;
          continue;
        }
        if (input.schemaIdentity == mapping::mappingArtifactSchema.identity &&
            input.schemaVersion == mapping::mappingArtifactSchema.version)
          techMappings.push_back(input);
      }
    if (!moduleReference || techMappings.empty())
      return invalid("SpatialMapping feedback lacks its exact Mapping inputs");
    canonicalizeRoots(techMappings);
    auto adopted = mapping::adoptSpatialGraphBoundaryEndpointHallFeedback(
        feedback.canonicalPayload, *moduleReference, techMappings, artifacts);
    if (!adopted)
      return adopted.takeError();
    SpatialHardwareFeedbackObservation candidate{std::move(*adopted)};
    if (!selected ||
        candidate.feedback.requiredBoundaryPairs() >
            selected->feedback.requiredBoundaryPairs() ||
        (candidate.feedback.requiredBoundaryPairs() ==
             selected->feedback.requiredBoundaryPairs() &&
         mapping::encodeSpatialGraphBoundaryEndpointHallFeedback(
             candidate.feedback) <
             mapping::encodeSpatialGraphBoundaryEndpointHallFeedback(
                 selected->feedback)))
      selected = std::move(candidate);
  }
  return selected;
}

llvm::Expected<std::optional<SystemHardwareFeedbackObservation>>
selectSystemHardwareFeedback(const dse::JointDesignExecution &execution,
                             const ArtifactStore &artifacts) {
  const dse::CompletedDsePlanExecution &available =
      availableExecution(execution.planExecution);
  std::optional<SystemHardwareFeedbackObservation> selected;
  for (const dse::GenerateInvocationFeedback &feedback :
       available.generateFeedback()) {
    const auto invocation = llvm::find_if(
        available.generateInvocations(), [&](const auto &candidate) {
          return candidate.planNodeOrdinal == feedback.planNodeOrdinal;
        });
    if (invocation == available.generateInvocations().end())
      return invalid("Generate feedback has no invocation owner");
    const dse::CandidateGeneratorDescriptor *descriptor =
        invocation->generatorBinding.descriptorRef().descriptor();
    if (!descriptor || !descriptor->ownerFeedbackPayload ||
        descriptor->ownerFeedbackPayload->schemaDescriptorBytes !=
            mapping::systemAccCoreCapacityPressureSchemaBytes())
      continue;

    std::optional<ArtifactRootReference> system;
    std::optional<ArtifactRootReference> dataflow;
    std::vector<ArtifactRootReference> spatialMappings;
    for (const dse::CandidateGeneratorInputBinding &binding :
         invocation->inputBindings)
      for (const ArtifactRootReference &input : binding.artifacts) {
        if (input.schemaIdentity == fabric::fabricArtifactSchema.identity &&
            input.schemaVersion == fabric::fabricArtifactSchema.version) {
          auto imported = fabric::importEntireFabricRoot(input, artifacts);
          if (!imported)
            return imported.takeError();
          if (imported->view().rootKind() != fabric::FabricRootKind::System)
            continue;
          if (system)
            return invalid(
                "SystemMapping feedback names multiple System inputs");
          system = input;
          continue;
        }
        if (input.schemaIdentity == mapping::mappingArtifactSchema.identity &&
            input.schemaVersion == mapping::mappingArtifactSchema.version)
          spatialMappings.push_back(input);
        if (input.schemaIdentity ==
                ::dataflow::canonicalDataflowSchema.identity &&
            input.schemaVersion ==
                ::dataflow::canonicalDataflowSchema.version) {
          if (dataflow)
            return invalid(
                "SystemMapping feedback names multiple Dataflow inputs");
          dataflow = input;
        }
      }
    if (!system || !dataflow || spatialMappings.empty())
      return invalid("SystemMapping feedback lacks its exact Mapping inputs");
    auto adopted = mapping::adoptSystemAccCoreCapacityPressure(
        feedback.canonicalPayload, *system, *dataflow, spatialMappings,
        artifacts);
    if (!adopted)
      return adopted.takeError();
    SystemHardwareFeedbackObservation candidate{std::move(*adopted)};
    const std::uint64_t candidateOveruse = candidate.feedback.witnessUsage() -
                                           candidate.feedback.witnessCapacity();
    const std::uint64_t selectedOveruse =
        selected ? selected->feedback.witnessUsage() -
                       selected->feedback.witnessCapacity()
                 : 0;
    if (!selected || candidateOveruse > selectedOveruse ||
        (candidateOveruse == selectedOveruse &&
         mapping::encodeSystemAccCoreCapacityPressure(candidate.feedback) <
             mapping::encodeSystemAccCoreCapacityPressure(selected->feedback)))
      selected = std::move(candidate);
  }
  return selected;
}

llvm::Expected<HardwareRecipeGrowth> deriveHardwareRecipeGrowth(
    const ResolvedConfig &baseConfig,
    const std::optional<TechHardwareFeedbackObservation> &techObservation,
    const std::optional<SpatialHardwareFeedbackObservation> &spatialObservation,
    const std::optional<SystemHardwareFeedbackObservation> &systemObservation,
    const ArtifactStore &artifacts) {
  HardwareRecipeGrowth growth;
  growth.config = baseConfig;
  growth.resultingContexts =
      baseConfig.hardwareTarget.parameters.temporalResidentContexts;
  growth.resultingGateways = baseConfig.hardwareTarget.parameters.gatewayCount;
  growth.resultingAccCores = baseConfig.hardwareTarget.parameters.accCoreCount;

  if (techObservation) {
    auto module =
        fabric::importEntireFabricRoot(techObservation->module, artifacts);
    if (!module)
      return module.takeError();
    auto plan = dse::projectTechMappingComputeContextJointGrowthPlan(
        techObservation->feedback, module->view());
    if (!plan)
      return plan.takeError();
    std::uint64_t requiredContexts = 0;
    for (const dse::ResizeInstructionStore &decision : plan->decisions) {
      const std::uint64_t currentCapacity =
          module->view().peResidentContextCount(decision.target);
      if (decision.instructionCapacity <= currentCapacity)
        return invalid("joint Module growth contains a non-growth decision");
      requiredContexts =
          std::max(requiredContexts,
                   static_cast<std::uint64_t>(decision.instructionCapacity));
    }
    const std::uint64_t currentContexts = growth.resultingContexts;
    if (requiredContexts <= currentContexts ||
        requiredContexts > std::numeric_limits<std::uint32_t>::max())
      return invalid("joint Module growth has no representable uniform "
                     "context growth");
    growth.addedContexts = requiredContexts - currentContexts;
    growth.resultingContexts = requiredContexts;
    growth.config.hardwareTarget.parameters.temporalResidentContexts =
        static_cast<std::uint32_t>(requiredContexts);
  }

  if (spatialObservation) {
    const std::uint64_t addedGateways =
        spatialObservation->feedback.requiredBoundaryPairs();
    if (addedGateways == 0 ||
        addedGateways > std::numeric_limits<std::uint32_t>::max() -
                            growth.resultingGateways)
      return invalid("graph-boundary Hall feedback has no representable "
                     "gateway growth");
    growth.addedGateways = addedGateways;
    growth.resultingGateways += addedGateways;
    growth.config.hardwareTarget.parameters.gatewayCount =
        static_cast<std::uint32_t>(growth.resultingGateways);
  }

  if (systemObservation) {
    if (systemObservation->feedback.compatibleAccCoreCount() !=
        growth.resultingAccCores)
      return invalid("uniform recipe cannot represent heterogeneous AccCore "
                     "capacity growth");
    if (growth.resultingAccCores == std::numeric_limits<std::uint32_t>::max())
      return invalid("AccCore capacity growth exceeds the builtin recipe");
    growth.addedAccCores = systemObservation->feedback.additionalAccCoreCount();
    growth.resultingAccCores += growth.addedAccCores;
    growth.config.hardwareTarget.parameters.accCoreCount =
        static_cast<std::uint32_t>(growth.resultingAccCores);
    growth.accCoreParent = systemObservation->feedback.system();
    growth.accCoreTargetModule = systemObservation->feedback.targetModule();
  }

  if (growth.addedContexts == 0 && growth.addedGateways == 0 &&
      growth.addedAccCores == 0)
    return invalid("Mapping feedback requests no builtin recipe growth");
  growth.config.dse.planNodes.clear();
  return growth;
}

llvm::Expected<MaterializedHardwareCandidate> materializeHardwareRecipeGrowth(
    HardwareRecipeGrowth growth, llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "mapping_feedback_recipe_growth";
        fields["added_temporal_contexts"] = growth.addedContexts;
        fields["temporal_resident_contexts"] = growth.resultingContexts;
        fields["added_gateways"] = growth.addedGateways;
        fields["gateway_count"] = growth.resultingGateways;
        fields["added_acc_cores"] = growth.addedAccCores;
        fields["acc_core_count"] = growth.resultingAccCores;
      });

  auto templateConfig =
      dse::projectResolvedFabricTemplateConfigView(growth.config);
  if (!templateConfig)
    return templateConfig.takeError();
  auto binding =
      dse::resolveFabricTemplateCandidateGeneratorBinding(*templateConfig);
  if (!binding)
    return binding.takeError();
  growth.config.dse.planNodes = {dse::GeneratePlanNodeDefinition{
      binding->descriptorRef(),
      {},
      templateConfig->canonicalViewBytes().vec(),
      templateConfig->digest()}};
  auto execution = executeResolvedGeneratePlan(
      growth.config, {}, evidence, request, scheduler, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  const dse::CompletedDsePlanExecution &available =
      availableExecution(*execution);
  const auto outputs = available.resolve({0, 0});
  if (outputs.size() != 1 || available.generateInvocations().size() != 1)
    return invalid("uniform recipe growth did not publish one System");
  auto system = fabric::importEntireFabricRoot(outputs.front(), artifacts);
  if (!system)
    return system.takeError();
  if (system->view().rootKind() != fabric::FabricRootKind::System)
    return invalid("uniform recipe growth published a non-System root");
  growth.config.dse.planNodes.clear();
  return MaterializedHardwareCandidate{outputs.front(),
                                       std::move(growth.config),
                                       std::nullopt,
                                       growth.addedContexts,
                                       growth.resultingContexts,
                                       growth.addedGateways,
                                       growth.resultingGateways,
                                       growth.addedAccCores,
                                       growth.resultingAccCores};
}

llvm::Expected<ArtifactRootReference>
accCoreTargetModule(const fabric::FinalizedFabricRoot &system,
                    const fabric::FabricSystemRootView &view,
                    fabric::AccCoreOccurrenceRef core) {
  const auto target = view.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= system.directDependencies().size())
    return invalid("AccCore has no exact imported Module target");
  const fabric::FabricDirectDependency &dependency =
      system.directDependencies()[target->dependencyOrdinal];
  if (dependency.role != fabric::FabricDependencyRole::ImportedModule)
    return invalid("AccCore target is not an ImportedModule dependency");
  return dependency.root;
}

llvm::Expected<MaterializedHardwareCandidate>
materializeTypedAccCoreGrowth(HardwareRecipeGrowth growth,
                              const ArtifactStore &artifacts,
                              const BlobStore &blobs) {
  if (growth.addedContexts != 0 || growth.addedGateways != 0 ||
      growth.addedAccCores != 1 || !growth.accCoreParent ||
      !growth.accCoreTargetModule)
    return invalid("typed AddAccCore materialization received a mixed change");
  auto parent =
      fabric::importEntireFabricRoot(*growth.accCoreParent, artifacts);
  if (!parent)
    return parent.takeError();
  auto parentView = fabric::requireSystemRoot(parent->view());
  if (!parentView)
    return parentView.takeError();
  std::optional<fabric::AccCoreOccurrenceRef> prototype;
  for (fabric::AccCoreOccurrenceRef core :
       parentView->artifact().accCoreOccurrences()) {
    auto target = accCoreTargetModule(*parent, *parentView, core);
    if (!target)
      return target.takeError();
    if (*target == *growth.accCoreTargetModule) {
      prototype = core;
      break;
    }
  }
  if (!prototype)
    return invalid("typed AddAccCore growth has no compatible prototype");

  std::vector<SystemCompositionDecisionDomain> domains = {
      AddAccCoreDomain{*prototype, {*growth.accCoreTargetModule}}};
  auto config = resolveSystemCompositionRewriteConfig(domains, 1);
  if (!config)
    return config.takeError();
  auto inputs = bindSystemCompositionCandidateGeneratorInputs(
      {*growth.accCoreParent}, {*growth.accCoreTargetModule});
  if (!inputs)
    return inputs.takeError();
  auto binding = resolveSystemCompositionCandidateGeneratorBinding(*config);
  if (!binding)
    return binding.takeError();
  auto generated =
      invokeCandidateGenerator(*inputs, *binding, artifacts, blobs);
  if (!generated)
    return generated.takeError();
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&generated->outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1 ||
      completed->lineageEdges.size() != 1)
    return invalid("typed AddAccCore growth did not publish one exact child");
  const ArtifactRootReference childReference =
      completed->outputBindings.front().artifacts.front();
  const CandidateGeneratorLineageEdge &lineage =
      completed->lineageEdges.front();
  if (lineage.kind != CandidateGeneratorLineageEdgeKind::CandidateDecision ||
      lineage.output != childReference ||
      lineage.parents !=
          std::vector<ArtifactRootReference>{*growth.accCoreParent})
    return invalid("typed AddAccCore child lost its CandidateDecision lineage");
  auto decision = adoptSystemCompositionDecision(lineage.ownerPayload);
  if (!decision)
    return decision.takeError();
  const auto *added = std::get_if<AddAccCore>(&decision->decision);
  if (decision->parent != *growth.accCoreParent || !added ||
      added->prototype != *prototype ||
      added->module != *growth.accCoreTargetModule)
    return invalid("typed AddAccCore lineage changed its exact decision");
  auto child = fabric::importEntireFabricRoot(childReference, artifacts);
  if (!child)
    return child.takeError();
  std::vector<pnr::SystemAccCoreCorrespondence> pairs;
  pairs.reserve(decision->accCoreCorrespondence.size());
  for (const auto &entry : decision->accCoreCorrespondence)
    pairs.push_back({entry.parent, entry.child});
  auto correspondence = pnr::SystemExecutionBindingCorrespondence::get(
      parent->reference(), child->reference(), std::move(pairs), artifacts);
  if (!correspondence)
    return correspondence.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "typed_add_acc_core_growth";
        fields["preserved_acc_core_correspondences"] =
            correspondence->accCores().size();
        fields["added_acc_cores"] = growth.addedAccCores;
      });
  growth.config.dse.planNodes.clear();
  return MaterializedHardwareCandidate{childReference,
                                       std::move(growth.config),
                                       std::move(*correspondence),
                                       growth.addedContexts,
                                       growth.resultingContexts,
                                       growth.addedGateways,
                                       growth.resultingGateways,
                                       growth.addedAccCores,
                                       growth.resultingAccCores};
}

llvm::Expected<std::vector<ArtifactRootReference>>
normalizedTimingProfiles(const ArtifactRootReference &system,
                         const ArtifactStore &artifacts) {
  auto modules = dse::projectJointDesignTargetModules(system, artifacts);
  if (!modules)
    return modules.takeError();
  std::vector<ArtifactRootReference> profiles;
  profiles.reserve(modules->size());
  for (const ArtifactRootReference &moduleReference : *modules) {
    auto module = fabric::importEntireFabricRoot(moduleReference, artifacts);
    if (!module)
      return module.takeError();
    auto timing =
        fabric::projectNormalizedFabricPhysicalTimingProfile(module->view());
    if (!timing)
      return timing.takeError();
    auto reference =
        fabric::publishFabricPhysicalTimingProfile(*timing, artifacts);
    if (!reference)
      return reference.takeError();
    profiles.push_back(std::move(*reference));
  }
  return profiles;
}

llvm::Expected<std::optional<dse::JointDesignExecution>>
tryHardwareFeedbackReopen(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const dse::JointDesignExecution &failedExecution,
    std::optional<dse::JointDesignExecution> &lastFailedExecution,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (policy.maximumSystemFrontier() <= 1)
    return std::optional<dse::JointDesignExecution>{};
  if (plan.frontier.systemFrontier.size() != 1 ||
      plan.frontier.softwareFrontier.size() != 1)
    return invalid("application hardware reopen requires one exact pair");
  const dse::JointSoftwareScope &software =
      plan.frontier.softwareFrontier.front();
  auto reopenPolicy = dse::JointDesignPolicy::get(
      1, 1, 1, policy.maximumSpatialMappingsPerPair());
  if (!reopenPolicy)
    return reopenPolicy.takeError();

  ResolvedConfig currentConfig = plan.resolvedConfig;
  currentConfig.dse.planNodes.clear();
  const std::uint64_t parentContexts =
      currentConfig.hardwareTarget.parameters.temporalResidentContexts;
  const std::uint64_t parentGateways =
      currentConfig.hardwareTarget.parameters.gatewayCount;
  const std::uint64_t parentAccCores =
      currentConfig.hardwareTarget.parameters.accCoreCount;
  const dse::JointDesignExecution *currentFailure = &failedExecution;
  const dse::JointDesignExplorationPlan *currentPlan = &plan;
  std::optional<dse::JointDesignExecution> latestFailed;
  std::optional<dse::JointDesignExplorationPlan> latestFailedPlan;
  std::optional<std::vector<ArtifactRootReference>> reusableSpatialMappings;
  const std::uint64_t candidateLimit = policy.maximumSystemFrontier() - 1;
  for (std::uint64_t candidateOrdinal = 0; candidateOrdinal != candidateLimit;
       ++candidateOrdinal) {
    auto techObservation =
        selectTechHardwareFeedback(*currentFailure, artifacts);
    if (!techObservation)
      return techObservation.takeError();
    auto spatialObservation =
        selectSpatialHardwareFeedback(*currentFailure, artifacts);
    if (!spatialObservation)
      return spatialObservation.takeError();
    auto systemObservation =
        selectSystemHardwareFeedback(*currentFailure, artifacts);
    if (!systemObservation)
      return systemObservation.takeError();
    if (!*techObservation && !*spatialObservation && !*systemObservation)
      break;

    auto growth = deriveHardwareRecipeGrowth(currentConfig, *techObservation,
                                             *spatialObservation,
                                             *systemObservation, artifacts);
    if (!growth)
      return growth.takeError();
    const bool accCoreOnlyGrowth = growth->addedAccCores != 0 &&
                                   growth->addedContexts == 0 &&
                                   growth->addedGateways == 0;
    auto system = accCoreOnlyGrowth ? materializeTypedAccCoreGrowth(
                                          std::move(*growth), artifacts, blobs)
                                    : materializeHardwareRecipeGrowth(
                                          std::move(*growth), evidence, request,
                                          scheduler, artifacts, blobs);
    if (!system)
      return system.takeError();
    auto timing = normalizedTimingProfiles(system->reference, artifacts);
    if (!timing)
      return timing.takeError();
    auto reopenPlan = dse::buildJointDesignExplorationPlan(
        {{software.workloads}, {system->reference}}, *timing, *reopenPolicy,
        system->config, artifacts);
    if (!reopenPlan)
      return reopenPlan.takeError();
    if (accCoreOnlyGrowth) {
      if (!reusableSpatialMappings) {
        auto resolved =
            resolveSpatialMappingFrontier(*currentPlan, *currentFailure);
        if (!resolved)
          return resolved.takeError();
        reusableSpatialMappings = std::move(*resolved);
      }
      if (llvm::Error error = bindImmutableSpatialMappingFrontier(
              *reopenPlan, *reusableSpatialMappings, artifacts))
        return std::move(error);
      if (!*systemObservation || !system->executionBindingCorrespondence)
        return invalid("typed AddAccCore reopen lost its Mapping checkpoint or "
                       "parent-to-child correspondence");
      auto migrationSeed = pnr::finalizeSystemMappingCheckpointMigrationSeed(
          (*systemObservation)->feedback.executionBindingCheckpoint(),
          *system->executionBindingCorrespondence,
          (*systemObservation)->feedback.witnessAccCore(), artifacts);
      if (!migrationSeed)
        return migrationSeed.takeError();
      if (llvm::Error error = bindSystemMappingMigrationSeed(
              *reopenPlan, migrationSeed->reference(), artifacts))
        return std::move(error);
    } else {
      reusableSpatialMappings.reset();
    }
    auto execution = executeJointPlan(*reopenPlan, evidence, request, scheduler,
                                      artifacts, blobs);
    if (!execution)
      return execution.takeError();
    const std::size_t systemMappingCount = mappingCount(*execution);
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "hardware_reopen_mapping_attempt";
          fields["candidate_ordinal"] = candidateOrdinal;
          fields["added_temporal_contexts"] =
              system->resultingContexts - parentContexts;
          fields["temporal_resident_contexts"] = system->resultingContexts;
          fields["added_gateways"] = system->resultingGateways - parentGateways;
          fields["gateway_count"] = system->resultingGateways;
          fields["added_acc_cores"] =
              system->resultingAccCores - parentAccCores;
          fields["acc_core_count"] = system->resultingAccCores;
          fields["system"] =
              formatArtifactIdentityHex(system->reference.artifact);
          fields["system_mapping_count"] = systemMappingCount;
        });
    if (systemMappingCount != 0)
      return std::optional<dse::JointDesignExecution>{std::move(*execution)};
    if (const auto *incomplete =
            std::get_if<IncompleteDsePlanExecution>(&execution->planExecution);
        incomplete && incomplete->executionStopped())
      return std::optional<dse::JointDesignExecution>{std::move(*execution)};

    currentConfig = std::move(system->config);
    latestFailed = std::move(*execution);
    latestFailedPlan = std::move(*reopenPlan);
    currentFailure = &*latestFailed;
    currentPlan = &*latestFailedPlan;
  }
  if (latestFailed)
    lastFailedExecution = std::move(*latestFailed);
  return std::optional<dse::JointDesignExecution>{};
}

} // namespace

llvm::Expected<JointDesignExecution> executeJointDesignWithHardwareReopen(
    llvm::ArrayRef<const JointDesignExplorationPlan *> plans,
    const JointDesignPolicy &policy, JointHardwareReopenRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (llvm::Error error = registerProductionDseOwners())
    return std::move(error);
  if (request.journalRoot.empty())
    return invalid("hardware reopen requires a journal root");
  if (plans.empty())
    return invalid("hardware reopen requires at least one Mapping plan");
  auto scheduler = SiteScheduler::create(std::move(request.siteCapacity));
  if (!scheduler)
    return scheduler.takeError();
  loom::pnr::PnrDerivedContextSession derivedContextSession;
  struct FailedSoftwareAttempt final {
    const JointDesignExplorationPlan *plan = nullptr;
    JointDesignExecution execution;
  };
  std::vector<FailedSoftwareAttempt> failedSoftwareAttempts;
  failedSoftwareAttempts.reserve(plans.size());
  std::optional<JointDesignExecution> firstIncomplete;
  std::optional<JointDesignExecution> lastNoFeasible;
  for (const JointDesignExplorationPlan *planPointer : plans) {
    if (!planPointer)
      return invalid("hardware reopen plan pointer is null");
    const JointDesignExplorationPlan &plan = *planPointer;
    auto initial = executeJointPlan(plan, request.evidence, request, *scheduler,
                                    artifacts, blobs);
    if (!initial)
      return initial.takeError();
    if (mappingCount(*initial) != 0)
      return std::move(*initial);
    if (const auto *incomplete =
            std::get_if<IncompleteDsePlanExecution>(&initial->planExecution);
        incomplete && incomplete->executionStopped())
      return std::move(*initial);
    failedSoftwareAttempts.push_back({planPointer, std::move(*initial)});
  }
  // Hardware feedback is consumed only after every bounded software/System
  // pair has been tried on the parent System. This preserves the declared
  // software frontier order and prevents repairable early failures from
  // hiding a later parent-hardware solution.
  for (FailedSoftwareAttempt &attempt : failedSoftwareAttempts) {
    std::optional<JointDesignExecution> lastReopenedFailure;
    auto reopened = tryHardwareFeedbackReopen(
        policy, *attempt.plan, attempt.execution, lastReopenedFailure,
        request.evidence, request, *scheduler, artifacts, blobs);
    if (!reopened)
      return reopened.takeError();
    if (*reopened)
      return std::move(**reopened);
    JointDesignExecution &failed =
        lastReopenedFailure ? *lastReopenedFailure : attempt.execution;
    if (std::holds_alternative<IncompleteDsePlanExecution>(
            failed.planExecution)) {
      if (!firstIncomplete)
        firstIncomplete = std::move(failed);
    } else {
      lastNoFeasible = std::move(failed);
    }
  }
  if (firstIncomplete)
    return std::move(*firstIncomplete);
  return std::move(*lastNoFeasible);
}

} // namespace loom::dse
