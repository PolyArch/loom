#include "DSE/JointHardwareReopen.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/ProductionOwners.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "DSE/TechMappingHardwareFeedback.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"
#include "PnR/PnrDerivedContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <map>
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

struct RankedHardwareCandidate final {
  ArtifactRootReference reference;
  std::uint64_t addedContexts = 0;
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

std::vector<RankedHardwareCandidate>
boundedPortfolio(std::vector<RankedHardwareCandidate> candidates,
                 std::uint64_t limit) {
  llvm::sort(candidates, [](const auto &lhs, const auto &rhs) {
    if (lhs.addedContexts != rhs.addedContexts)
      return lhs.addedContexts < rhs.addedContexts;
    return artifactRootReferenceLess(lhs.reference, rhs.reference);
  });
  if (candidates.size() <= limit)
    return candidates;
  std::vector<RankedHardwareCandidate> selected;
  selected.reserve(static_cast<std::size_t>(limit));
  if (limit == 1) {
    selected.push_back(std::move(candidates.front()));
    return selected;
  }
  for (std::uint64_t ordinal = 0; ordinal != limit; ++ordinal) {
    const std::size_t index = static_cast<std::size_t>(
        ordinal * (candidates.size() - 1) / (limit - 1));
    selected.push_back(std::move(candidates[index]));
  }
  return selected;
}

llvm::Expected<RankedHardwareCandidate> generateJointModuleGrowth(
    const TechHardwareFeedbackObservation &observation,
    const ResolvedConfig &baseConfig,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto module = fabric::importEntireFabricRoot(observation.module, artifacts);
  if (!module)
    return module.takeError();
  auto plan = dse::projectTechMappingComputeContextJointGrowthPlan(
      observation.feedback, module->view());
  if (!plan)
    return plan.takeError();
  std::uint64_t maximumAddedContexts = 0;
  std::uint64_t maximumResultingCapacity = 0;
  for (const dse::ResizeInstructionStore &decision : plan->decisions) {
    const std::uint64_t currentCapacity =
        module->view().peResidentContextCount(decision.target);
    if (decision.instructionCapacity <= currentCapacity)
      return invalid("joint Module growth contains a non-growth decision");
    maximumAddedContexts =
        std::max(maximumAddedContexts,
                 static_cast<std::uint64_t>(decision.instructionCapacity) -
                     currentCapacity);
    maximumResultingCapacity =
        std::max(maximumResultingCapacity,
                 static_cast<std::uint64_t>(decision.instructionCapacity));
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::TechMapping,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "compute_context_joint_growth_decision";
          fields["pe"] = fabric::printFabricRef(decision.target);
          fields["current_contexts"] = currentCapacity;
          fields["resulting_contexts"] = decision.instructionCapacity;
          fields["added_contexts"] =
              static_cast<std::uint64_t>(decision.instructionCapacity) -
              currentCapacity;
        });
  }
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "compute_context_joint_growth";
        fields["hall_deficit"] = observation.feedback.deficit();
        fields["growth_pe_count"] = plan->decisions.size();
        fields["added_context_count"] = plan->addedContextCount;
        fields["maximum_pe_added_contexts"] = maximumAddedContexts;
        fields["maximum_resulting_contexts"] = maximumResultingCapacity;
      });
  auto rewriteConfig = dse::resolveSpatialMicroarchitectureRewriteConfig(
      {dse::ResizeInstructionStoresDomain{plan->decisions}}, 1);
  if (!rewriteConfig)
    return rewriteConfig.takeError();
  auto binding = dse::resolveSpatialMicroarchitectureCandidateGeneratorBinding(
      *rewriteConfig);
  if (!binding)
    return binding.takeError();
  ResolvedConfig config = baseConfig;
  config.dse.planNodes = {dse::GeneratePlanNodeDefinition{
      binding->descriptorRef(),
      {dse::ExactPlanArtifacts{{observation.module}}},
      rewriteConfig->canonicalViewBytes().vec(),
      rewriteConfig->digest()}};
  auto execution =
      executeResolvedGeneratePlan(config, {observation.module}, evidence,
                                  request, scheduler, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  const dse::CompletedDsePlanExecution &available =
      availableExecution(*execution);
  const auto outputs = available.resolve({0, 0});
  if (outputs.size() != 1 || available.generateInvocations().size() != 1 ||
      available.generateInvocations().front().lineageEdges.size() != 1)
    return invalid("joint Module growth did not publish one typed child");
  auto lineage =
      dse::adoptSpatialMicroarchitectureDecision(available.generateInvocations()
                                                     .front()
                                                     .lineageEdges.front()
                                                     .ownerPayload);
  if (!lineage)
    return lineage.takeError();
  const auto *applied =
      std::get_if<dse::ResizeInstructionStores>(&lineage->decision);
  if (!applied || applied->stores.size() != plan->decisions.size())
    return invalid("joint Module growth lineage changed its decision kind");
  for (auto [actual, expected] :
       llvm::zip_equal(applied->stores, plan->decisions))
    if (actual.target != expected.target ||
        actual.instructionCapacity != expected.instructionCapacity)
      return invalid("joint Module growth lineage changed a PE resize");
  if (outputs.front() == observation.module)
    return invalid("joint Module growth retained its parent identity");
  return RankedHardwareCandidate{outputs.front(), plan->addedContextCount};
}

llvm::Expected<std::vector<RankedHardwareCandidate>> generateModuleGrowth(
    const TechHardwareFeedbackObservation &observation,
    const ResolvedConfig &baseConfig, const dse::JointDesignPolicy &policy,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::vector<RankedHardwareCandidate> result;
  auto joint = generateJointModuleGrowth(observation, baseConfig, evidence,
                                         request, scheduler, artifacts, blobs);
  if (!joint)
    return joint.takeError();
  if (llvm::none_of(result, [&](const auto &candidate) {
        return candidate.reference == joint->reference;
      }))
    result.push_back(std::move(*joint));
  return boundedPortfolio(std::move(result), policy.maximumSystemFrontier());
}

llvm::Expected<std::vector<RankedHardwareCandidate>> generateSystemGrowth(
    const ArtifactRootReference &parentSystem,
    const ArtifactRootReference &parentModule,
    llvm::ArrayRef<RankedHardwareCandidate> modules,
    const ResolvedConfig &baseConfig, const dse::JointDesignPolicy &policy,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (modules.empty())
    return std::vector<RankedHardwareCandidate>{};
  auto systemArtifact = fabric::importEntireFabricRoot(parentSystem, artifacts);
  if (!systemArtifact)
    return systemArtifact.takeError();
  auto system = fabric::requireSystemRoot(systemArtifact->view());
  if (!system)
    return system.takeError();
  std::optional<fabric::AccCoreOccurrenceRef> target;
  for (fabric::AccCoreOccurrenceRef core :
       system->artifact().accCoreOccurrences()) {
    const auto selected = system->spatialCoreTarget(core);
    if (!selected || selected->dependencyOrdinal >=
                         system->artifact().importedModules().size())
      return invalid("System AccCore has no exact Module target");
    if (system->artifact()
            .importedModules()[selected->dependencyOrdinal]
            .identity() == parentModule.artifact) {
      target = core;
      break;
    }
  }
  if (!target)
    return invalid("System contains no AccCore for the feedback Module");
  std::vector<ArtifactRootReference> moduleReferences;
  moduleReferences.reserve(modules.size());
  std::map<ArtifactRootReference, std::uint64_t,
           decltype(&artifactRootReferenceLess)>
      moduleGrowth(&artifactRootReferenceLess);
  for (const RankedHardwareCandidate &module : modules) {
    moduleReferences.push_back(module.reference);
    moduleGrowth.emplace(module.reference, module.addedContexts);
  }
  std::vector<ArtifactRootReference> canonicalModules = moduleReferences;
  canonicalizeRoots(canonicalModules);
  auto rewriteConfig = dse::resolveSystemCompositionRewriteConfig(
      {dse::ReplaceSpatialAttachmentDomain{*target, canonicalModules}},
      canonicalModules.size());
  if (!rewriteConfig)
    return rewriteConfig.takeError();
  auto binding =
      dse::resolveSystemCompositionCandidateGeneratorBinding(*rewriteConfig);
  if (!binding)
    return binding.takeError();
  ResolvedConfig config = baseConfig;
  config.dse.planNodes = {dse::GeneratePlanNodeDefinition{
      binding->descriptorRef(),
      {dse::ExactPlanArtifacts{{parentSystem}},
       dse::ExactPlanArtifacts{canonicalModules}},
      rewriteConfig->canonicalViewBytes().vec(),
      rewriteConfig->digest()}};
  std::vector<ArtifactRootReference> semanticInputs{parentSystem};
  semanticInputs.insert(semanticInputs.end(), canonicalModules.begin(),
                        canonicalModules.end());
  auto execution =
      executeResolvedGeneratePlan(config, std::move(semanticInputs), evidence,
                                  request, scheduler, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  const dse::CompletedDsePlanExecution &available =
      availableExecution(*execution);
  const auto outputs = available.resolve({0, 0});
  if (outputs.empty())
    return std::vector<RankedHardwareCandidate>{};
  if (available.generateInvocations().size() != 1)
    return invalid("System growth plan has the wrong invocation count");
  std::map<ArtifactRootReference, std::uint64_t,
           decltype(&artifactRootReferenceLess)>
      growthBySystem(&artifactRootReferenceLess);
  for (const dse::CandidateGeneratorLineageEdge &edge :
       available.generateInvocations().front().lineageEdges) {
    auto decoded = dse::adoptSystemCompositionDecision(edge.ownerPayload);
    if (!decoded)
      return decoded.takeError();
    const auto *replace =
        std::get_if<dse::ReplaceSpatialAttachment>(&decoded->decision);
    if (!replace)
      return invalid("System growth lineage contains a foreign decision");
    const auto growth = moduleGrowth.find(replace->module);
    if (growth == moduleGrowth.end())
      return invalid("System growth selected an unknown Module child");
    auto [found, inserted] =
        growthBySystem.emplace(edge.output, growth->second);
    if (!inserted)
      found->second = std::min(found->second, growth->second);
  }
  std::vector<RankedHardwareCandidate> result;
  result.reserve(outputs.size());
  for (const ArtifactRootReference &output : outputs) {
    const auto growth = growthBySystem.find(output);
    if (growth == growthBySystem.end())
      return invalid("System growth output has no typed decision lineage");
    result.push_back({output, growth->second});
  }
  return boundedPortfolio(std::move(result), policy.maximumSystemFrontier());
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
tryHardwareFeedbackReopen(const JointDesignPolicy &policy,
                          const JointDesignExplorationPlan &plan,
                          const dse::JointDesignExecution &failedExecution,
                          llvm::ArrayRef<ArtifactRootReference> evidence,
                          const JointHardwareReopenRequest &request,
                          dse::SiteScheduler &scheduler,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  if (policy.maximumSystemFrontier() <= 1)
    return std::optional<dse::JointDesignExecution>{};
  auto observation = selectTechHardwareFeedback(failedExecution, artifacts);
  if (!observation)
    return observation.takeError();
  if (!*observation)
    return std::optional<dse::JointDesignExecution>{};
  if (plan.frontier.systemFrontier.size() != 1 ||
      plan.frontier.softwareFrontier.size() != 1)
    return invalid("application hardware reopen requires one exact pair");
  ResolvedConfig baseConfig = plan.resolvedConfig;
  baseConfig.dse.planNodes.clear();
  auto modules =
      generateModuleGrowth(**observation, baseConfig, policy, evidence, request,
                           scheduler, artifacts, blobs);
  if (!modules)
    return modules.takeError();
  auto systems = generateSystemGrowth(
      plan.frontier.systemFrontier.front(), (*observation)->module, *modules,
      baseConfig, policy, evidence, request, scheduler, artifacts, blobs);
  if (!systems)
    return systems.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "compute_context_hardware_reopen";
        fields["hall_deficit"] = (*observation)->feedback.deficit();
        fields["module_candidate_count"] = modules->size();
        fields["system_candidate_count"] = systems->size();
      });

  const dse::JointSoftwareScope &software =
      plan.frontier.softwareFrontier.front();
  for (const auto indexed : llvm::enumerate(*systems)) {
    const std::size_t ordinal = indexed.index();
    const RankedHardwareCandidate &system = indexed.value();
    auto timing = normalizedTimingProfiles(system.reference, artifacts);
    if (!timing)
      return timing.takeError();
    auto reopenPolicy = dse::JointDesignPolicy::get(
        1, 1, 1, policy.maximumSpatialMappingsPerPair());
    if (!reopenPolicy)
      return reopenPolicy.takeError();
    auto plan = dse::buildJointDesignExplorationPlan(
        {{software.workloads}, {system.reference}}, *timing, *reopenPolicy,
        baseConfig, artifacts);
    if (!plan)
      return plan.takeError();
    auto execution =
        executeJointPlan(*plan, evidence, request, scheduler, artifacts, blobs);
    if (!execution)
      return execution.takeError();
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "hardware_reopen_mapping_attempt";
          fields["candidate_ordinal"] = ordinal;
          fields["added_temporal_contexts"] = system.addedContexts;
          fields["system"] =
              formatArtifactIdentityHex(system.reference.artifact);
          fields["system_mapping_count"] = mappingCount(*execution);
        });
    if (mappingCount(*execution) != 0)
      return std::optional<dse::JointDesignExecution>{std::move(*execution)};
  }
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
    auto reopened = tryHardwareFeedbackReopen(
        policy, *attempt.plan, attempt.execution, request.evidence, request,
        *scheduler, artifacts, blobs);
    if (!reopened)
      return reopened.takeError();
    if (*reopened)
      return std::move(**reopened);
    if (std::holds_alternative<IncompleteDsePlanExecution>(
            attempt.execution.planExecution)) {
      if (!firstIncomplete)
        firstIncomplete = std::move(attempt.execution);
    } else {
      lastNoFeasible = std::move(attempt.execution);
    }
  }
  if (firstIncomplete)
    return std::move(*firstIncomplete);
  return std::move(*lastNoFeasible);
}

} // namespace loom::dse
