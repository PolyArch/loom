#include "DSE/JointDesignExploration.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Evidence.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "PnR/PnrConfig.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_design_exploration_invalid: " + message);
}

llvm::Expected<ResolvedConfig>
importInvocationResolvedConfig(const ArtifactRootReference &reference,
                               const ArtifactStore &artifacts) {
  if (reference.schemaIdentity != ResolvedConfig::artifactSchema.identity ||
      reference.schemaVersion != ResolvedConfig::artifactSchema.version)
    return invalid("invocation manifest reference names a non-ResolvedConfig "
                   "Artifact");
  auto stored = artifacts.get(reference);
  if (!stored)
    return stored.takeError();
  const llvm::ArrayRef<std::uint8_t> bytes = stored->bytes();
  auto config = parseResolvedConfig(
      llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                      bytes.size()),
      "stored invocation ResolvedConfig");
  if (!config)
    return config.takeError();
  if (resolvedConfigIdentity(*config) != reference.artifact ||
      canonicalResolvedConfigBytes(*config).bytes() != bytes)
    return invalid("invocation ResolvedConfig is not its exact canonical "
                   "Artifact");
  return config;
}

llvm::Expected<InvocationManifest> adoptJointDesignInvocationManifest(
    const ArtifactRootReference &resolvedConfig, const BlobDigest &blob,
    const InvocationOccurrenceRef &occurrence, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  auto config = importInvocationResolvedConfig(resolvedConfig, artifacts);
  if (!config)
    return config.takeError();
  auto canonical = blobs.get(blob);
  if (!canonical)
    return canonical.takeError();
  auto manifest = adoptInvocationManifest(*canonical, *config, artifacts);
  if (!manifest)
    return manifest.takeError();
  if (manifest->occurrence() != occurrence)
    return invalid("invocation manifest reference occurrence differs from "
                   "the canonical Blob");
  return manifest;
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Error registerMappingGenerators() {
  if (llvm::Error error =
          registerApplicationGraphTechMappingCandidateGenerator())
    return error;
  if (llvm::Error error = registerRootCompleteSpatialPnrCandidateGenerator())
    return error;
  if (llvm::Error error = registerSpatialPnrCandidateGenerator())
    return error;
  return registerApplicationSystemPnrCandidateGenerator();
}

llvm::Expected<std::vector<ArtifactRootReference>>
projectTargetModules(const ArtifactRootReference &systemReference,
                     const ArtifactStore &store) {
  auto artifact = fabric::importEntireFabricRoot(systemReference, store);
  if (!artifact)
    return artifact.takeError();
  auto system = fabric::requireSystemRoot(artifact->view());
  if (!system)
    return system.takeError();

  std::vector<ArtifactRootReference> modules;
  for (fabric::AccCoreOccurrenceRef core :
       system->artifact().accCoreOccurrences()) {
    auto target = system->spatialCoreTarget(core);
    if (!target || target->dependencyOrdinal >=
                       system->artifact().importedModules().size())
      return invalid("System AccCore has no exact imported Module target");
    const fabric::FabricArtifactView &module =
        system->artifact().importedModules()[target->dependencyOrdinal];
    if (!module.moduleRootTemplate() ||
        *module.moduleRootTemplate() != target->target)
      return invalid("System AccCore target is not its imported Module root");
    modules.push_back({fabric::fabricArtifactSchema.identity.str(),
                       fabric::fabricArtifactSchema.version,
                       module.identity()});
  }
  llvm::sort(modules, artifactRootReferenceLess);
  modules.erase(std::unique(modules.begin(), modules.end()), modules.end());
  if (modules.empty())
    return invalid("System has no Mapping target Module");
  for (const ArtifactRootReference &module : modules) {
    auto imported = fabric::importEntireFabricRoot(module, store);
    if (!imported)
      return imported.takeError();
    if (!imported->view().moduleRootTemplate())
      return invalid("System target dependency is not a Module root");
  }
  return modules;
}

llvm::Expected<std::vector<dataflow::RootThreadLaunchRef>>
scopeRoots(const JointSoftwareScope &scope, const ArtifactStore &store) {
  std::vector<dataflow::RootThreadLaunchRef> roots;
  roots.reserve(scope.workloads.size());
  for (const ArtifactRootReference &reference : scope.workloads) {
    auto imported = sim::importSpatialSimulationWorkload(reference, store);
    if (!imported)
      return imported.takeError();
    if (imported->dataflow.identity() != scope.dataflow.artifact)
      return invalid("application workload has a foreign Dataflow owner");
    const sim::SpatialSimulationWorkload *workload =
        imported->workload.spatial();
    if (!workload)
      return invalid("application scope contains a non-Spatial workload");
    roots.push_back(workload->launchRef.rootThreadLaunch);
  }
  llvm::sort(roots, [](const auto &lhs, const auto &rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  if (roots.empty())
    return invalid("application scope projects no root thread launch");
  return roots;
}

llvm::Expected<ArtifactRootReference>
publishScopeConstraints(const JointSoftwareScope &scope,
                        const ArtifactRootReference &systemReference,
                        const ArtifactStore &store) {
  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(scope.dataflow, store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto systemArtifact = fabric::importEntireFabricRoot(systemReference, store);
  if (!systemArtifact)
    return systemArtifact.takeError();
  auto system = fabric::requireSystemRoot(systemArtifact->view());
  if (!system)
    return system.takeError();
  auto roots = scopeRoots(scope, store);
  if (!roots)
    return roots.takeError();
  auto constraints = mapping::finalizeEmptySystemMappingConstraintSet(
      *dataflow, *system, *roots, store);
  if (!constraints)
    return constraints.takeError();
  return constraints->reference();
}

void canonicalize(std::vector<ArtifactRootReference> &roots) {
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

bool rootsAreCanonical(llvm::ArrayRef<ArtifactRootReference> roots) {
  return llvm::is_sorted(roots, artifactRootReferenceLess) &&
         std::adjacent_find(roots.begin(), roots.end()) == roots.end();
}

llvm::Error
validateEvidenceRoots(llvm::ArrayRef<ArtifactRootReference> evidence,
                      const ArtifactStore &artifactStore) {
  if (!rootsAreCanonical(evidence))
    return invalid("member Promotion Evidence is not canonical and unique");
  for (const ArtifactRootReference &reference : evidence) {
    if (reference.schemaIdentity !=
            evaluation::EvaluationEvidence::artifactSchema.identity ||
        reference.schemaVersion !=
            evaluation::EvaluationEvidence::artifactSchema.version)
      return invalid("member Promotion retained a non-Evidence root");
    auto stored = artifactStore.get(reference);
    if (!stored)
      return stored.takeError();
  }
  return llvm::Error::success();
}

struct ImportedSystem final {
  ArtifactRootReference reference;
  fabric::FinalizedFabricRoot artifact;
  fabric::FabricSystemRootView view;
  std::vector<std::vector<ArtifactRootReference>> memberMappings;
  std::set<std::string> usedAccCores;
};

llvm::Expected<std::size_t> findSystem(llvm::ArrayRef<ImportedSystem> systems,
                                       const ArtifactIdentity &identity) {
  for (std::size_t index = 0; index != systems.size(); ++index)
    if (systems[index].reference.artifact == identity)
      return index;
  return invalid("accepted SystemMapping has a foreign System owner");
}

} // namespace

llvm::Expected<JointDesignInvocationManifestReference>
JointDesignInvocationManifestReference::get(
    ArtifactRootReference resolvedConfig, BlobDigest blob,
    InvocationOccurrenceRef occurrence, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  auto manifest = adoptJointDesignInvocationManifest(
      resolvedConfig, blob, occurrence, artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  return JointDesignInvocationManifestReference(
      std::move(resolvedConfig), std::move(blob), std::move(occurrence));
}

llvm::Expected<JointDesignInvocationManifestReference>
publishJointDesignInvocationManifest(const InvocationManifest &manifest,
                                     const ResolvedConfig &resolvedConfig,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs) {
  auto configIdentity =
      artifacts.put(ResolvedConfig::artifactSchema,
                    canonicalResolvedConfigBytes(resolvedConfig));
  if (!configIdentity)
    return configIdentity.takeError();
  if (*configIdentity != resolvedConfigIdentity(resolvedConfig) ||
      *configIdentity != manifest.closure().resolvedConfigIdentity())
    return invalid("invocation manifest names a different ResolvedConfig");
  auto adopted = adoptInvocationManifest(manifest.canonicalBytes(),
                                         resolvedConfig, artifacts);
  if (!adopted)
    return adopted.takeError();
  auto blob = blobs.put(manifest.canonicalBytes());
  if (!blob)
    return blob.takeError();
  return JointDesignInvocationManifestReference::get(
      ArtifactRootReference{ResolvedConfig::artifactSchema.identity.str(),
                            ResolvedConfig::artifactSchema.version,
                            std::move(*configIdentity)},
      std::move(*blob), manifest.occurrence(), artifacts, blobs);
}

llvm::Expected<InvocationManifest> importJointDesignInvocationManifest(
    const JointDesignInvocationManifestReference &reference,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return adoptJointDesignInvocationManifest(
      reference.resolvedConfig(), reference.blob(), reference.occurrence(),
      artifacts, blobs);
}

JointDesignQualityDisposition jointDesignQualityDisposition(
    JointDesignQualityIncompleteReason reason) {
  switch (reason) {
  case JointDesignQualityIncompleteReason::Unsupported:
    return JointDesignQualityDisposition::Unsupported;
  case JointDesignQualityIncompleteReason::ProofNotEstablished:
    return JointDesignQualityDisposition::ProofNotEstablished;
  case JointDesignQualityIncompleteReason::ExecutionFailed:
    return JointDesignQualityDisposition::ExecutionFailed;
  case JointDesignQualityIncompleteReason::CancelledOrTimeout:
    return JointDesignQualityDisposition::CancelledOrTimeout;
  }
  llvm_unreachable("unknown bounded-quality incomplete reason");
}

llvm::Error validateJointDesignQualityProvenanceDomain(
    const JointBoundedQualityPolicy &policy,
    const JointDesignQualityProvenance &provenance, bool objectiveComplete) {
  switch (policy.provenanceDomain) {
  case JointDesignQualityProvenanceDomain::ObjectiveOnly:
    return llvm::Error::success();
  case JointDesignQualityProvenanceDomain::ApplicationRuntime:
    break;
  }
  if (policy.objectiveDimensionLabels.size() < 3 ||
      policy.objectiveDimensionLabels[0] != "dfg_cycles" ||
      policy.objectiveDimensionLabels[1] != "cgra_cycles" ||
      policy.objectiveDimensionLabels[2] != "acc_core_count")
    return invalid("ApplicationRuntime provenance has a foreign Objective "
                   "domain");
  if (!provenance.resourceCoreCost)
    return invalid("ApplicationRuntime provenance lost its exact resource "
                   "count");
  if (provenance.rawMeasures.empty() && objectiveComplete)
    return invalid("complete ApplicationRuntime provenance lost its raw "
                   "measures");
  if (provenance.rawMeasures.empty())
    return llvm::Error::success();
  if (provenance.rawMeasures.size() < 3)
    return invalid("ApplicationRuntime provenance has incomplete raw "
                   "measures");
  const auto *dfg =
      std::get_if<ResolvedObjectiveInteger>(&provenance.rawMeasures[0]);
  const auto *cgra =
      std::get_if<ResolvedObjectiveInteger>(&provenance.rawMeasures[1]);
  const auto *cores =
      std::get_if<ResolvedObjectiveInteger>(&provenance.rawMeasures[2]);
  if (!dfg || dfg->negative || !cgra || cgra->negative || !cores ||
      cores->negative)
    return invalid("ApplicationRuntime provenance has invalid runtime raw "
                   "measures");
  if (*provenance.resourceCoreCost != cores->magnitude)
    return invalid("ApplicationRuntime resource count disagrees with its raw "
                   "measure");
  return llvm::Error::success();
}

llvm::Error validateJointDesignQualityObjective(
    const ObjectiveProgram &program,
    const JointDesignQualityProvenance &provenance,
    llvm::ArrayRef<std::uint64_t> objectiveCodes) {
  if (provenance.rawMeasures.empty())
    return llvm::Error::success();
  ObjectiveVector reproduced = program.makeVector();
  if (llvm::Error error =
          program.evaluateCandidateMeasures(provenance.rawMeasures, reproduced))
    return error;
  if (reproduced.codes() != objectiveCodes)
    return invalid("quality provenance raw measures disagree with its "
                   "Objective codes");
  return llvm::Error::success();
}

llvm::Expected<std::vector<ArtifactRootReference>>
projectJointDesignTargetModules(const ArtifactRootReference &system,
                                const ArtifactStore &artifactStore) {
  return projectTargetModules(system, artifactStore);
}

llvm::Expected<JointDesignExplorationPlan> buildJointDesignExplorationPlan(
    JointDesignInputs inputs,
    llvm::ArrayRef<ArtifactRootReference> physicalTimingProfiles,
    const JointDesignPolicy &policy, const ResolvedConfig &baseConfig,
    const ArtifactStore &artifactStore,
    const JointDesignMappingSeed *mappingSeed,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> systemBindingPartitions) {
  if (!baseConfig.dse.planNodes.empty())
    return invalid("base ResolvedConfig already owns a DSE invocation plan");
  if (llvm::Error error = registerMappingGenerators())
    return std::move(error);
  auto frontier =
      buildBoundedJointFrontier(std::move(inputs), policy, artifactStore);
  if (!frontier)
    return frontier.takeError();
  auto techConfig = mapping::projectResolvedTechMappingConfigView(baseConfig);
  if (!techConfig)
    return techConfig.takeError();
  auto spatialConfig = pnr::projectResolvedSpatialPnrConfigView(baseConfig);
  if (!spatialConfig)
    return spatialConfig.takeError();
  auto projectedSystemConfig =
      pnr::projectResolvedSystemPnrConfigView(baseConfig);
  if (!projectedSystemConfig)
    return projectedSystemConfig.takeError();
  pnr::ResolvedPnrConfigView systemConfig = std::move(*projectedSystemConfig);
  if (!systemBindingPartitions.empty()) {
    auto specialized = pnr::specializeResolvedSystemPnrConfigView(
        systemConfig, systemBindingPartitions);
    if (!specialized)
      return specialized.takeError();
    systemConfig = std::move(*specialized);
  }

  std::map<ArtifactIdentity::Storage, ArtifactRootReference> timingByModule;
  for (const ArtifactRootReference &profile : physicalTimingProfiles) {
    auto owner =
        fabric::resolveFabricPhysicalTimingProfileOwner(profile, artifactStore);
    if (!owner)
      return owner.takeError();
    if (!timingByModule.emplace(owner->bytes(), profile).second)
      return invalid(
          "multiple physical timing profiles target the same Module");
  }

  struct SeedMapping final {
    ArtifactRootReference reference;
    ArtifactIdentity dataflow;
    ArtifactIdentity module;
    bool matched = false;
  };
  std::vector<SeedMapping> seedTechMappings;
  std::vector<SeedMapping> seedSpatialMappings;
  struct SeedRepairConstraint final {
    ArtifactRootReference techMapping;
    ArtifactRootReference constraintSet;
    ArtifactIdentity dataflow;
    ArtifactIdentity module;
    bool matched = false;
  };
  std::vector<SeedRepairConstraint> seedRepairConstraints;
  if (mappingSeed) {
    seedTechMappings.reserve(mappingSeed->techMappings.size());
    for (const ArtifactRootReference &reference : mappingSeed->techMappings) {
      auto imported = mapping::importTechMapping(reference, artifactStore);
      if (!imported)
        return imported.takeError();
      seedTechMappings.push_back({reference,
                                  imported->view().dataflowIdentity(),
                                  imported->view().fabricIdentity(), false});
    }
    seedSpatialMappings.reserve(mappingSeed->spatialMappings.size());
    for (const ArtifactRootReference &reference :
         mappingSeed->spatialMappings) {
      auto imported = mapping::importSpatialMapping(reference, artifactStore);
      if (!imported)
        return imported.takeError();
      seedSpatialMappings.push_back({reference,
                                     imported->view().dataflowIdentity(),
                                     imported->view().fabricIdentity(), false});
    }
    seedRepairConstraints.reserve(mappingSeed->spatialRepairConstraints.size());
    for (const auto &repair : mappingSeed->spatialRepairConstraints) {
      auto tech = mapping::importTechMapping(repair.techMapping, artifactStore);
      if (!tech)
        return tech.takeError();
      auto constraints = mapping::importSpatialMappingConstraintSet(
          repair.constraintSet, artifactStore);
      if (!constraints)
        return constraints.takeError();
      if (constraints->view().techMappingIdentity() !=
              tech->view().identity() ||
          constraints->view().dataflowIdentity() !=
              tech->view().dataflowIdentity() ||
          constraints->view().fabricIdentity() != tech->view().fabricIdentity())
        return invalid("Spatial repair constraint has inconsistent owners");
      seedRepairConstraints.push_back({repair.techMapping, repair.constraintSet,
                                       tech->view().dataflowIdentity(),
                                       tech->view().fabricIdentity(), false});
    }
  }

  ResolvedConfig planConfig = baseConfig;
  planConfig.dse.planNodes.clear();
  std::vector<JointDesignPlanPair> outputs;
  outputs.reserve(frontier->pairs.size());
  for (const JointDesignPair &pair : frontier->pairs) {
    auto constraints =
        publishScopeConstraints(pair.software, pair.system, artifactStore);
    if (!constraints)
      return constraints.takeError();
    auto modules = projectJointDesignTargetModules(pair.system, artifactStore);
    if (!modules)
      return modules.takeError();
    if (modules->size() > policy.maximumSpatialMappingsPerPair())
      return invalid("SpatialMapping join bound is smaller than the System's "
                     "distinct target Module count");
    std::vector<ArtifactRootReference> moduleTimingProfiles;
    moduleTimingProfiles.reserve(modules->size());
    for (const ArtifactRootReference &module : *modules) {
      const auto profile = timingByModule.find(module.artifact.bytes());
      if (profile == timingByModule.end())
        return invalid(
            "joint System target Module has no physical timing profile");
      auto importedModule =
          fabric::importEntireFabricRoot(module, artifactStore);
      if (!importedModule)
        return importedModule.takeError();
      auto importedProfile = fabric::importFabricPhysicalTimingProfile(
          profile->second, importedModule->view(), artifactStore);
      if (!importedProfile)
        return importedProfile.takeError();
      moduleTimingProfiles.push_back(profile->second);
    }
    std::vector<ArtifactRootReference> systemTimingProfiles =
        moduleTimingProfiles;
    canonicalize(systemTimingProfiles);
    std::vector<PlanOutputRef> techOutputs;
    techOutputs.reserve(modules->size());
    std::vector<PlanOutputRef> spatialOutputs;
    spatialOutputs.reserve(modules->size());
    std::vector<ArtifactRootReference> immutableTechMappings;
    std::vector<ArtifactRootReference> immutableSpatialMappings;
    for (std::size_t moduleIndex = 0; moduleIndex != modules->size();
         ++moduleIndex) {
      const ArtifactRootReference &module = (*modules)[moduleIndex];
      std::vector<ArtifactRootReference> retainedTech;
      std::vector<ArtifactRootReference> retainedSpatial;
      for (SeedMapping &seed : seedTechMappings)
        if (seed.dataflow == pair.software.dataflow.artifact &&
            seed.module == module.artifact) {
          retainedTech.push_back(seed.reference);
          seed.matched = true;
        }
      for (SeedMapping &seed : seedSpatialMappings)
        if (seed.dataflow == pair.software.dataflow.artifact &&
            seed.module == module.artifact) {
          retainedSpatial.push_back(seed.reference);
          seed.matched = true;
        }
      canonicalize(retainedTech);
      canonicalize(retainedSpatial);
      SeedRepairConstraint *repairConstraint = nullptr;
      for (SeedRepairConstraint &candidate : seedRepairConstraints) {
        if (candidate.dataflow != pair.software.dataflow.artifact ||
            candidate.module != module.artifact)
          continue;
        if (repairConstraint)
          return invalid("joint pair has multiple Spatial repair constraints");
        repairConstraint = &candidate;
      }
      if (retainedTech.size() > policy.maximumTechMappingsPerModule())
        return invalid("immutable TechMapping frontier exceeds its bound");
      immutableTechMappings.insert(immutableTechMappings.end(),
                                   retainedTech.begin(), retainedTech.end());
      immutableSpatialMappings.insert(immutableSpatialMappings.end(),
                                      retainedSpatial.begin(),
                                      retainedSpatial.end());
      if (!retainedSpatial.empty()) {
        if (repairConstraint)
          return invalid("Spatial repair constraint also retains an immutable "
                         "SpatialMapping");
        continue;
      }

      if (repairConstraint) {
        if (retainedTech.size() != 1 ||
            retainedTech.front() != repairConstraint->techMapping)
          return invalid("Spatial repair constraint requires its one exact "
                         "TechMapping seed");
        repairConstraint->matched = true;
        const std::uint64_t spatialNode = planConfig.dse.planNodes.size();
        planConfig.dse.planNodes.push_back(GeneratePlanNodeDefinition{
            spatialPnrCandidateGeneratorDescriptor().reference(),
            {ExactPlanArtifacts{{pair.software.dataflow}},
             ExactPlanArtifacts{{repairConstraint->techMapping}},
             ExactPlanArtifacts{{module}},
             ExactPlanArtifacts{{moduleTimingProfiles[moduleIndex]}},
             ExactPlanArtifacts{{repairConstraint->constraintSet}}},
            spatialConfig->canonicalViewBytes().vec(),
            spatialConfig->digest()});
        spatialOutputs.push_back(PlanOutputRef{spatialNode, 0});
        continue;
      }

      PlanInputBinding spatialTechInput;
      if (!retainedTech.empty()) {
        spatialTechInput = ExactPlanArtifacts{std::move(retainedTech)};
      } else {
        const std::uint64_t techNode = planConfig.dse.planNodes.size();
        planConfig.dse.planNodes.push_back(GeneratePlanNodeDefinition{
            applicationGraphTechMappingCandidateGeneratorDescriptor()
                .reference(),
            {ExactPlanArtifacts{{pair.software.dataflow}},
             ExactPlanArtifacts{{*constraints}}, ExactPlanArtifacts{{module}}},
            techConfig->canonicalViewBytes().vec(),
            techConfig->digest()});
        const PlanOutputRef techOutput{techNode, 0};
        techOutputs.push_back(techOutput);
        spatialTechInput = BoundedPlanOutputJoin{
            {techOutput}, policy.maximumTechMappingsPerModule()};
      }
      const std::uint64_t spatialNode = planConfig.dse.planNodes.size();
      planConfig.dse.planNodes.push_back(GeneratePlanNodeDefinition{
          rootCompleteSpatialPnrCandidateGeneratorDescriptor().reference(),
          {std::move(spatialTechInput), ExactPlanArtifacts{{module}},
           ExactPlanArtifacts{{moduleTimingProfiles[moduleIndex]}}},
          spatialConfig->canonicalViewBytes().vec(),
          spatialConfig->digest()});
      spatialOutputs.push_back(PlanOutputRef{spatialNode, 0});
    }
    canonicalize(immutableTechMappings);
    canonicalize(immutableSpatialMappings);
    if (immutableSpatialMappings.size() >
        policy.maximumSpatialMappingsPerPair())
      return invalid("immutable SpatialMapping frontier exceeds its bound");
    const std::uint64_t systemNode = planConfig.dse.planNodes.size();
    const std::vector<PlanOutputRef> retainedSpatialOutputs = spatialOutputs;
    PlanInputBinding systemSpatialInput;
    if (spatialOutputs.empty()) {
      systemSpatialInput = ExactPlanArtifacts{immutableSpatialMappings};
    } else {
      systemSpatialInput = BoundedPlanOutputJoin{
          std::move(spatialOutputs), policy.maximumSpatialMappingsPerPair(), 0,
          immutableSpatialMappings};
    }
    planConfig.dse.planNodes.push_back(GeneratePlanNodeDefinition{
        applicationSystemPnrCandidateGeneratorDescriptor().reference(),
        {ExactPlanArtifacts{{pair.software.dataflow}},
         std::move(systemSpatialInput), ExactPlanArtifacts{{pair.system}},
         ExactPlanArtifacts{std::move(systemTimingProfiles)},
         ExactPlanArtifacts{{*constraints}}, ExactPlanArtifacts{},
         ExactPlanArtifacts{}},
        systemConfig.canonicalViewBytes().vec(),
        systemConfig.digest()});
    outputs.push_back({pair, std::move(techOutputs), retainedSpatialOutputs,
                       std::move(immutableTechMappings),
                       std::move(immutableSpatialMappings),
                       PlanOutputRef{systemNode, 0}});
  }
  if (llvm::any_of(seedTechMappings,
                   [](const SeedMapping &seed) { return !seed.matched; }) ||
      llvm::any_of(seedSpatialMappings,
                   [](const SeedMapping &seed) { return !seed.matched; }) ||
      llvm::any_of(seedRepairConstraints, [](const SeedRepairConstraint &seed) {
        return !seed.matched;
      }))
    return invalid("immutable Mapping seed has no exact joint pair owner");
  auto admitted = projectResolvedDseConfigView(planConfig);
  if (!admitted)
    return admitted.takeError();
  return JointDesignExplorationPlan{
      std::move(planConfig), std::move(*frontier), std::move(outputs),
      systemConfig.systemBindingPartitions().vec()};
}

std::vector<ArtifactRootReference>
projectJointDesignSemanticInputs(const JointDesignExplorationPlan &plan) {
  std::vector<ArtifactRootReference> inputs;
  for (const JointSoftwareScope &scope : plan.frontier.softwareFrontier) {
    inputs.push_back(scope.dataflow);
    inputs.insert(inputs.end(), scope.workloads.begin(), scope.workloads.end());
  }
  inputs.insert(inputs.end(), plan.frontier.systemFrontier.begin(),
                plan.frontier.systemFrontier.end());
  for (const DsePlanNodeDefinition &node : plan.resolvedConfig.dse.planNodes) {
    const auto &bindings = std::visit(
        [](const auto &definition) -> const std::vector<PlanInputBinding> & {
          return definition.inputBindings;
        },
        node);
    for (const PlanInputBinding &binding : bindings) {
      if (const auto *exact = std::get_if<ExactPlanArtifacts>(&binding)) {
        inputs.insert(inputs.end(), exact->artifacts.begin(),
                      exact->artifacts.end());
      } else if (const auto *join =
                     std::get_if<BoundedPlanOutputJoin>(&binding)) {
        inputs.insert(inputs.end(), join->exactArtifacts.begin(),
                      join->exactArtifacts.end());
      }
    }
  }
  canonicalize(inputs);
  return inputs;
}

llvm::Expected<JointDesignExecution> executeJointDesignExploration(
    const JointDesignExplorationPlan &plan, const DseRunClosure &closure,
    ExecutionJournal &journal, SiteScheduler &scheduler,
    const PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (closure.resolvedConfigIdentity() !=
      resolvedConfigIdentity(plan.resolvedConfig))
    return invalid("run closure does not bind the joint plan ResolvedConfig");
  auto view = projectResolvedDseConfigView(plan.resolvedConfig);
  if (!view)
    return view.takeError();
  const auto executionStart = std::chrono::steady_clock::now();
  auto execution = resumeDsePlan(*view, closure, journal, scheduler,
                                 executionPolicy, artifactStore, blobStore,
                                 InvocationManifestRetention::Retain);
  const auto executionElapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - executionStart)
          .count();
  if (!execution)
    return execution.takeError();

  const CompletedDsePlanExecution *completed =
      std::get_if<CompletedDsePlanExecution>(&*execution);
  if (!completed)
    completed =
        &std::get<IncompleteDsePlanExecution>(*execution).availableExecution();
  std::vector<JointMappedPair> mappedPairs;
  for (const JointDesignPlanPair &pair : plan.pairOutputs) {
    if (!completed->hasOutput(pair.systemMappings))
      continue;
    std::vector<ArtifactRootReference> mappings =
        completed->resolve(pair.systemMappings).vec();
    for (const ArtifactRootReference &reference : mappings) {
      auto imported = mapping::importSystemMapping(reference, artifactStore);
      if (!imported)
        return imported.takeError();
      if (imported->view().dataflowIdentity() !=
              pair.pair.software.dataflow.artifact ||
          imported->view().fabricIdentity() != pair.pair.system.artifact)
        return invalid("joint plan output has foreign Dataflow/System owners");
    }
    mappedPairs.push_back({pair.pair, std::move(mappings)});
  }
  JointDesignExecutionSummary summary;
  summary.eligibleJointPairCount = plan.frontier.eligiblePairCount;
  summary.analyticEvaluatedJointPairCount =
      plan.frontier.analyticEvaluatedPairCount;
  summary.analyticDeferredJointPairCount =
      plan.frontier.analyticDeferredPairCount;
  summary.retainedJointPairCount = plan.frontier.pairs.size();
  summary.jointFrontierTruncated = plan.frontier.truncated;
  summary.retainedJointPairAnalytics.reserve(plan.frontier.pairs.size());
  for (std::size_t index = 0; index != plan.frontier.pairs.size(); ++index)
    summary.retainedJointPairAnalytics.push_back(
        {plan.frontier.pairs[index].software.dataflow,
         plan.frontier.pairs[index].system,
         plan.frontier.pairProjections[index]});
  const CandidateGeneratorDescriptorRef systemGenerator =
      applicationSystemPnrCandidateGeneratorDescriptor().reference();
  const CandidateGeneratorDescriptorRef spatialGenerator =
      rootCompleteSpatialPnrCandidateGeneratorDescriptor().reference();
  const CandidateGeneratorDescriptorRef constrainedSpatialGenerator =
      spatialPnrCandidateGeneratorDescriptor().reference();
  const CandidateGeneratorDescriptorRef techGenerator =
      applicationGraphTechMappingCandidateGeneratorDescriptor().reference();
  for (auto indexed : llvm::enumerate(completed->generateInvocations())) {
    const CandidateGeneratorDescriptorRef descriptor =
        indexed.value().generatorBinding.descriptorRef();
    const bool dispatched =
        completed->generateInvocationWasDispatched(indexed.index());
    if (descriptor == techGenerator) {
      ++summary.techMappingInvocationCount;
      if (dispatched)
        ++summary.techMappingDispatchCount;
      else
        ++summary.techMappingJournalReplayCount;
    } else if (descriptor == spatialGenerator ||
               descriptor == constrainedSpatialGenerator) {
      ++summary.spatialPnrInvocationCount;
      if (dispatched)
        ++summary.spatialPnrDispatchCount;
      else
        ++summary.spatialPnrJournalReplayCount;
    } else if (descriptor == systemGenerator) {
      ++summary.systemPnrInvocationCount;
      if (dispatched)
        ++summary.systemPnrDispatchCount;
      else
        ++summary.systemPnrJournalReplayCount;
    }
  }
  summary.executionWallTimeNanoseconds =
      static_cast<std::uint64_t>(std::max<std::int64_t>(0, executionElapsed));
  return JointDesignExecution{std::move(*execution), std::move(mappedPairs),
                              std::move(summary)};
}

llvm::Expected<JointDesignSelectionOutcome> selectJointDesignSystems(
    llvm::ArrayRef<ArtifactRootReference> systemRoots,
    llvm::ArrayRef<JointMemberPromotion> memberPromotions,
    llvm::ArrayRef<CandidateObjectiveVector> systemObjectives,
    const CandidateSelectionPolicy &selection,
    const ObjectiveProgram *objectiveProgram,
    const ArtifactStore &artifactStore) {
  if (systemRoots.empty() || memberPromotions.empty())
    return invalid("selection requires nonempty System and member sets");
  std::vector<ArtifactRootReference> canonicalSystems(systemRoots.begin(),
                                                      systemRoots.end());
  llvm::sort(canonicalSystems, artifactRootReferenceLess);
  if (std::adjacent_find(canonicalSystems.begin(), canonicalSystems.end()) !=
      canonicalSystems.end())
    return invalid("selection System set contains a duplicate root");

  std::vector<const JointMemberPromotion *> members;
  members.reserve(memberPromotions.size());
  for (const JointMemberPromotion &member : memberPromotions)
    members.push_back(&member);
  llvm::sort(members, [](const JointMemberPromotion *lhs,
                         const JointMemberPromotion *rhs) {
    return artifactRootReferenceLess(lhs->software, rhs->software);
  });
  for (std::size_t index = 1; index != members.size(); ++index)
    if (members[index - 1]->software == members[index]->software)
      return invalid("selection contains duplicate workload members");

  std::vector<ImportedSystem> systems;
  systems.reserve(canonicalSystems.size());
  for (const ArtifactRootReference &reference : canonicalSystems) {
    auto artifact = fabric::importEntireFabricRoot(reference, artifactStore);
    if (!artifact)
      return artifact.takeError();
    auto view = fabric::requireSystemRoot(artifact->view());
    if (!view)
      return view.takeError();
    if (view->artifact().accCoreOccurrences().empty())
      return invalid("selection System has no AccCore occurrence");
    systems.push_back(
        {reference,
         std::move(*artifact),
         std::move(*view),
         std::vector<std::vector<ArtifactRootReference>>(members.size()),
         {}});
  }

  std::vector<ArtifactRootReference> satisfiedEvidence;
  for (std::size_t memberIndex = 0; memberIndex != members.size();
       ++memberIndex) {
    const JointMemberPromotion &member = *members[memberIndex];
    auto dataflowArtifact =
        dataflow::importCanonicalDataflow(member.software, artifactStore);
    if (!dataflowArtifact)
      return dataflowArtifact.takeError();
    auto dataflow = dataflowArtifact->view();
    if (!dataflow)
      return dataflow.takeError();
    if (member.promotion.selected.empty() ||
        !rootsAreCanonical(member.promotion.selected))
      return invalid("member Promotion selected set is not nonempty and "
                     "canonical");
    if (llvm::Error error = validateEvidenceRoots(
            member.promotion.satisfiedEvidence, artifactStore))
      return std::move(error);
    satisfiedEvidence.insert(satisfiedEvidence.end(),
                             member.promotion.satisfiedEvidence.begin(),
                             member.promotion.satisfiedEvidence.end());

    for (const ArtifactRootReference &reference : member.promotion.selected) {
      auto systemMapping =
          mapping::importSystemMapping(reference, artifactStore);
      if (!systemMapping)
        return systemMapping.takeError();
      if (systemMapping->view().dataflowIdentity() != member.software.artifact)
        return invalid("member Promotion selected a foreign Dataflow Mapping");
      auto systemIndex =
          findSystem(systems, systemMapping->view().fabricIdentity());
      if (!systemIndex)
        return systemIndex.takeError();
      systems[*systemIndex].memberMappings[memberIndex].push_back(reference);
      auto contexts = mapping::projectSystemExecutionContexts(
          *dataflow, systemMapping->view().executionBindings());
      if (!contexts)
        return contexts.takeError();
      for (const mapping::SystemInstructionContextDomain &domain :
           contexts->instructionDomains) {
        systems[*systemIndex].usedAccCores.insert(
            byteKey(fabric::canonicalFabricBytes(domain.context.accCore)));
      }
    }
  }
  canonicalize(satisfiedEvidence);

  std::vector<JointSystemGateOutcome> outcomes;
  std::vector<ArtifactRootReference> eligibleSystems;
  outcomes.reserve(systems.size());
  for (ImportedSystem &system : systems) {
    std::optional<std::size_t> missing;
    for (std::size_t member = 0; member != members.size(); ++member)
      if (system.memberMappings[member].empty()) {
        missing = member;
        break;
      }
    if (missing) {
      outcomes.emplace_back(JointSystemMissingMember{
          system.reference, members[*missing]->software});
      continue;
    }

    std::optional<fabric::AccCoreOccurrenceRef> unused;
    for (fabric::AccCoreOccurrenceRef core :
         system.view.artifact().accCoreOccurrences())
      if (!system.usedAccCores.count(
              byteKey(fabric::canonicalFabricBytes(core)))) {
        unused = core;
        break;
      }
    if (unused) {
      outcomes.emplace_back(
          JointSystemUnusedAccCore{system.reference, *unused});
      continue;
    }

    std::vector<ArtifactRootReference> accepted;
    for (std::vector<ArtifactRootReference> &memberMappings :
         system.memberMappings)
      accepted.insert(accepted.end(), memberMappings.begin(),
                      memberMappings.end());
    canonicalize(accepted);
    eligibleSystems.push_back(system.reference);
    outcomes.emplace_back(
        JointEligibleSystem{system.reference, std::move(accepted)});
  }

  if (eligibleSystems.empty())
    return JointDesignSelectionOutcome{JointDesignNoFeasibleSystem{
        std::move(satisfiedEvidence), std::move(outcomes)}};
  auto candidateSet =
      CandidateSet::get(fabric::fabricArtifactSchema, canonicalSystems);
  if (!candidateSet)
    return candidateSet.takeError();
  auto selected =
      applyCandidateSelection(*candidateSet, eligibleSystems, systemObjectives,
                              selection, objectiveProgram);
  if (!selected)
    return selected.takeError();
  if (selected->empty())
    return JointDesignSelectionOutcome{JointDesignNoFeasibleSystem{
        std::move(satisfiedEvidence), std::move(outcomes)}};

  std::vector<ArtifactRootReference> acceptedMappings;
  for (const JointSystemGateOutcome &outcome : outcomes) {
    const auto *eligible = std::get_if<JointEligibleSystem>(&outcome);
    if (!eligible || !llvm::binary_search(*selected, eligible->system,
                                          artifactRootReferenceLess))
      continue;
    acceptedMappings.insert(acceptedMappings.end(),
                            eligible->acceptedMappings.begin(),
                            eligible->acceptedMappings.end());
  }
  canonicalize(acceptedMappings);
  return JointDesignSelectionOutcome{
      JointDesignSelection{std::move(*selected), std::move(acceptedMappings),
                           std::move(satisfiedEvidence), std::move(outcomes)}};
}

} // namespace loom::dse
