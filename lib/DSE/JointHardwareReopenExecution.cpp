#include "JointHardwareReopenExecution.h"

#include "JointHardwareReopenInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/Evidence.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <system_error>

namespace loom::dse {

class JointDesignExecutionManifestBinder final {
public:
  static llvm::Error
  bind(JointDesignExecution &execution,
       JointDesignInvocationManifestReference invocationManifest) {
    if (execution.invocationManifest_)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "joint execution already has an invocation manifest");
    execution.invocationManifest_.emplace(std::move(invocationManifest));
    return llvm::Error::success();
  }

  static llvm::Error
  appendSupporting(JointDesignExecution &execution,
                   JointDesignInvocationManifestReference invocationManifest) {
    const auto sameReference =
        [&](const JointDesignInvocationManifestReference &other) {
          return other.resolvedConfig() ==
                     invocationManifest.resolvedConfig() &&
                 other.blob() == invocationManifest.blob() &&
                 other.occurrence() == invocationManifest.occurrence();
        };
    const auto sameOccurrence =
        [&](const JointDesignInvocationManifestReference &other) {
          return other.occurrence() == invocationManifest.occurrence();
        };
    if (execution.invocationManifest_ &&
        sameOccurrence(*execution.invocationManifest_)) {
      if (!sameReference(*execution.invocationManifest_))
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "one DSE occurrence has conflicting manifests");
      return llvm::Error::success();
    }
    auto existing =
        llvm::find_if(execution.supportingInvocationManifests_, sameOccurrence);
    if (existing != execution.supportingInvocationManifests_.end()) {
      if (!sameReference(*existing))
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "one DSE occurrence has conflicting manifests");
      return llvm::Error::success();
    }
    execution.supportingInvocationManifests_.push_back(
        std::move(invocationManifest));
    return llvm::Error::success();
  }
};

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_hardware_reopen_execution_invalid: " + message);
}

std::vector<ArtifactRootReference>
mappingRoots(const JointDesignExecution &execution) {
  std::vector<ArtifactRootReference> roots;
  for (const JointMappedPair &pair : execution.mappedPairs)
    roots.insert(roots.end(), pair.systemMappings.begin(),
                 pair.systemMappings.end());
  std::sort(roots.begin(), roots.end(), artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  return roots;
}

struct RetainedInvocationRoots final {
  std::vector<ArtifactRootReference> artifacts;
  std::vector<ArtifactRootReference> evidence;
};

RetainedInvocationRoots
retainedInvocationRoots(const DsePlanGenerateInvocationRecords &records,
                        llvm::ArrayRef<ArtifactRootReference> mappings) {
  RetainedInvocationRoots retained;
  retained.artifacts.assign(mappings.begin(), mappings.end());
  const auto append = [&](llvm::ArrayRef<GenerateInvocationRecord> generated) {
    for (const GenerateInvocationRecord &record : generated)
      for (const CandidateGeneratorOutputBinding &binding :
           record.outputBindings)
        for (const ArtifactRootReference &root : binding.artifacts) {
          if (root.schemaIdentity ==
                  evaluation::EvaluationEvidence::artifactSchema.identity &&
              root.schemaVersion ==
                  evaluation::EvaluationEvidence::artifactSchema.version)
            retained.evidence.push_back(root);
          else
            retained.artifacts.push_back(root);
        }
  };
  append(records.completed());
  append(records.incomplete());
  const auto canonicalize = [](std::vector<ArtifactRootReference> &roots) {
    std::sort(roots.begin(), roots.end(), artifactRootReferenceLess);
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  };
  canonicalize(retained.artifacts);
  canonicalize(retained.evidence);
  return retained;
}

InvocationControllerOutcome
projectInvocationOutcome(const JointDesignExecution &execution,
                         const DsePlanGenerateInvocationRecords &records) {
  std::vector<ArtifactRootReference> mappings = mappingRoots(execution);
  if (const auto *incomplete =
          std::get_if<IncompleteDsePlanExecution>(&execution.planExecution)) {
    RetainedInvocationRoots retained =
        retainedInvocationRoots(records, mappings);
    return InvocationIncomplete{incomplete->nodeOrdinal(),
                                incomplete->reason(),
                                {},
                                std::move(retained.artifacts),
                                std::move(retained.evidence)};
  }
  if (mappings.empty())
    return InvocationCompletedNoFeasibleCandidate{};
  return InvocationCompletedSelection{std::move(mappings), {}};
}

llvm::Expected<std::optional<ArtifactRootReference>>
finalizeJointRepairSelection(JointDesignExecution &execution,
                             JointDesignStoppingPolicy stoppingPolicy);

} // namespace

llvm::Expected<JointDesignInvocationManifestReference>
publishJointPlanInvocationManifest(
    DseRunClosure closure, const ResolvedConfig &config,
    const DsePlanGenerateInvocationRecords &generateRecords,
    InvocationControllerOutcome outcome, ExecutionJournal &journal,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto occurrence = journal.currentInvocationOccurrence();
  if (!occurrence)
    return occurrence.takeError();
  auto externalToolWork = journal.externalToolWorkLedger();
  if (!externalToolWork)
    return externalToolWork.takeError();
  auto manifest = InvocationManifest::get(
      std::move(closure), occurrence->first.occurrenceOrdinal,
      std::move(occurrence->second), config, generateRecords,
      std::move(outcome), artifacts, std::nullopt,
      std::move(*externalToolWork));
  if (!manifest)
    return manifest.takeError();
  auto reference =
      publishJointDesignInvocationManifest(*manifest, config, artifacts, blobs);
  if (!reference)
    return reference.takeError();
  if (llvm::Error error = journal.commitInvocationManifest(
          reference->occurrence(), reference->blob()))
    return std::move(error);
  return reference;
}

llvm::Error bindJointDesignInvocationManifest(
    JointDesignExecution &execution,
    JointDesignInvocationManifestReference invocationManifest) {
  return JointDesignExecutionManifestBinder::bind(
      execution, std::move(invocationManifest));
}

llvm::Error appendJointDesignSupportingInvocationManifest(
    JointDesignExecution &execution,
    JointDesignInvocationManifestReference invocationManifest) {
  return JointDesignExecutionManifestBinder::appendSupporting(
      execution, std::move(invocationManifest));
}

llvm::Error retainJointDesignInvocationManifest(
    std::vector<JointDesignInvocationManifestReference> &retained,
    const JointDesignInvocationManifestReference &invocationManifest) {
  auto existing = llvm::find_if(retained, [&](const auto &candidate) {
    return candidate.occurrence() == invocationManifest.occurrence();
  });
  if (existing == retained.end()) {
    retained.push_back(invocationManifest);
    return llvm::Error::success();
  }
  if (existing->resolvedConfig() != invocationManifest.resolvedConfig() ||
      existing->blob() != invocationManifest.blob())
    return invalid("one DSE occurrence has conflicting manifests");
  return llvm::Error::success();
}

llvm::Error retainJointDesignExecutionInvocations(
    std::vector<JointDesignInvocationManifestReference> &retained,
    const JointDesignExecution &execution) {
  if (execution.invocationManifest())
    if (llvm::Error error = retainJointDesignInvocationManifest(
            retained, *execution.invocationManifest()))
      return error;
  for (const JointDesignInvocationManifestReference &supporting :
       execution.supportingInvocationManifests())
    if (llvm::Error error =
            retainJointDesignInvocationManifest(retained, supporting))
      return error;
  return llvm::Error::success();
}

llvm::Error attachJointDesignSupportingInvocationManifests(
    JointDesignExecution &execution,
    llvm::ArrayRef<JointDesignInvocationManifestReference> retained) {
  for (const JointDesignInvocationManifestReference &supporting : retained)
    if (llvm::Error error = appendJointDesignSupportingInvocationManifest(
            execution, supporting))
      return error;
  return llvm::Error::success();
}

llvm::Expected<JointDesignExecution>
executeJointPlan(const JointDesignExplorationPlan &plan,
                 llvm::ArrayRef<ArtifactRootReference> evidence,
                 const JointHardwareReopenRequest &request,
                 SiteScheduler &scheduler, const ArtifactStore &artifacts,
                 const BlobStore &blobs,
                 const PlanExecutionPolicy *executionPolicy) {
  const ResolvedConfig &config = plan.resolvedConfig;
  auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                       canonicalResolvedConfigBytes(config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(config))
    return invalid("ResolvedConfig publication changed its identity");
  std::vector<ArtifactRootReference> semanticInputs =
      projectJointDesignSemanticInputs(plan);
  semanticInputs.insert(semanticInputs.end(),
                        request.invocationSemanticInputs.begin(),
                        request.invocationSemanticInputs.end());
  std::sort(semanticInputs.begin(), semanticInputs.end(),
            artifactRootReferenceLess);
  semanticInputs.erase(
      std::unique(semanticInputs.begin(), semanticInputs.end()),
      semanticInputs.end());
  auto closure = DseRunClosure::get(request.producer, semanticInputs, config,
                                    evidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto configView = projectResolvedDseConfigView(config);
  if (!configView)
    return configView.takeError();
  llvm::SmallString<256> journalPath(request.journalRoot);
  llvm::sys::path::append(journalPath,
                          llvm::toHex(closure->runKey().bytes(), true));
  if (std::error_code error = llvm::sys::fs::create_directories(journalPath))
    return invalid("cannot create Mapping alternative journal: " +
                   error.message());
  auto journal = openExecutionJournal(journalPath, *closure, *configView);
  if (!journal)
    return journal.takeError();
  auto execution = executeJointDesignExploration(
      plan, *closure, *journal, scheduler,
      executionPolicy ? *executionPolicy : request.executionPolicy, artifacts,
      blobs);
  if (!execution)
    return execution.takeError();
  DsePlanGenerateInvocationRecords records =
      projectDsePlanGenerateInvocationRecords(execution->planExecution);
  auto manifest = publishJointPlanInvocationManifest(
      std::move(*closure), config, records,
      projectInvocationOutcome(*execution, records), *journal, artifacts,
      blobs);
  if (!manifest)
    return manifest.takeError();
  if (llvm::Error error =
          bindJointDesignInvocationManifest(*execution, std::move(*manifest)))
    return std::move(error);
  return execution;
}

llvm::Expected<JointDesignExecution>
executeJointRepairPlan(const JointDesignExplorationPlan &plan,
                       const JointDesignPolicy &policy,
                       JointHardwareReopenRequest request,
                       const ArtifactStore &artifacts, const BlobStore &blobs) {
  const JointDesignStoppingPolicy stoppingPolicy = request.stoppingPolicy;
  llvm::Expected<JointDesignExecution> execution = [&]()
      -> llvm::Expected<JointDesignExecution> {
    if (request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality) {
      if (!request.boundedQuality)
        return invalid("bounded repair has no quality policy");
      request.hardwareExplorationScope =
          JointHardwareExplorationScope::FixedSystemFrontier;
      const JointDesignExplorationPlan *planPointer = &plan;
      return executeJointDesignWithHardwareReopen(
          llvm::ArrayRef<const JointDesignExplorationPlan *>(&planPointer, 1),
          policy, std::move(request), artifacts, blobs);
    }
    if (request.boundedQuality)
      return invalid("first-verified repair carries a quality policy");
    auto scheduler = SiteScheduler::create(request.siteCapacity);
    if (!scheduler)
      return scheduler.takeError();
    return executeJointPlan(plan, request.evidence, request, *scheduler,
                            artifacts, blobs);
  }();
  if (!execution)
    return execution.takeError();
  auto selected = finalizeJointRepairSelection(*execution, stoppingPolicy);
  if (!selected)
    return selected.takeError();
  return std::move(*execution);
}

namespace {

llvm::Expected<std::optional<ArtifactRootReference>>
finalizeJointRepairSelection(JointDesignExecution &execution,
                             JointDesignStoppingPolicy stoppingPolicy) {
  if (stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality) {
    if (execution.summary.qualityDisposition !=
        JointDesignQualityDisposition::Complete) {
      execution.summary.selectedMapping.reset();
      execution.summary.selectedPlanOrdinal.reset();
      return std::nullopt;
    }
    if (!execution.summary.selectedMapping)
      return invalid("completed bounded repair has no selected Mapping");
    const std::vector<ArtifactRootReference> mappings =
        joint_reopen_detail::mappingRoots(execution);
    if (!llvm::is_contained(mappings, *execution.summary.selectedMapping))
      return invalid("bounded repair selected a Mapping outside its output");
    return execution.summary.selectedMapping;
  }

  std::optional<ArtifactRootReference> selected =
      joint_reopen_detail::firstMapping(execution);
  execution.summary.selectedMapping = selected;
  execution.summary.selectedPlanOrdinal =
      selected ? std::optional<std::uint64_t>(0) : std::nullopt;
  return selected;
}

} // namespace

llvm::Expected<std::vector<ArtifactRootReference>>
normalizedTimingProfiles(const ArtifactRootReference &system,
                         const ArtifactStore &artifacts) {
  auto modules = projectJointDesignTargetModules(system, artifacts);
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

} // namespace loom::dse
