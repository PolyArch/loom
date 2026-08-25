#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CampaignRunner.h"
#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/GroundTruthPlan.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/ModelParameterCalibrationAcquisition.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"
#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"
#include "DSE/ProductionOwners.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialMappingEvaluationAcquisition.h"
#include "DSE/SpatialMappingFeedbackCandidateGenerator.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SpatialTopologyCandidateGenerator.h"
#include "DSE/StructuredEvaluationAcquisition.h"
#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredSpecialMathAccuracyCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/LocalConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;

llvm::cl::opt<std::string>
    configPath("config", llvm::cl::Required,
               llvm::cl::desc("canonical resolved configuration JSON"),
               llvm::cl::value_desc("path"));
llvm::cl::opt<std::string> artifactStorePath(
    "artifact-store", llvm::cl::Required,
    llvm::cl::desc("existing content-addressed ArtifactStore root"),
    llvm::cl::value_desc("directory"));
llvm::cl::opt<std::string>
    blobStorePath("blob-store", llvm::cl::Required,
                  llvm::cl::desc("existing content-addressed BlobStore root"),
                  llvm::cl::value_desc("directory"));
llvm::cl::opt<std::string>
    runRoot("run-root", llvm::cl::Required,
            llvm::cl::desc("durable execution journal directory"),
            llvm::cl::value_desc("directory"));
llvm::cl::opt<std::string>
    producerBuild("producer-build", llvm::cl::Required,
                  llvm::cl::desc("semantic producer build identity"),
                  llvm::cl::value_desc("identity"));
llvm::cl::list<std::string> semanticInputFiles(
    "semantic-input",
    llvm::cl::desc("JSON file containing one exact semantic input root"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> evidenceInputFiles(
    "preexisting-evidence",
    llvm::cl::desc("JSON file containing one exact preexisting Evidence root"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> jointApplicationScopeFiles(
    "joint-application-scope",
    llvm::cl::desc("canonical Spatial workload-root set admitted as one "
                   "application scope"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> jointSystemRootFiles(
    "joint-system-root",
    llvm::cl::desc("exact Fabric System root admitted to joint Mapping"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> jointPhysicalTimingProfileFiles(
    "joint-physical-timing-profile",
    llvm::cl::desc("exact physical timing profile root for a joint target "
                   "Module"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::opt<bool> jointNormalizedPhysicalTiming(
    "joint-normalized-physical-timing",
    llvm::cl::desc("explicitly publish and bind target-neutral normalized "
                   "timing for every joint target Module"),
    llvm::cl::init(false));
llvm::cl::opt<std::uint64_t> jointPairLimit(
    "joint-pair-limit",
    llvm::cl::desc(
        "maximum joint software/System pairs; zero admits the full product"),
    llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> jointSpatialMappingLimit(
    "joint-spatial-mapping-limit",
    llvm::cl::desc("maximum SpatialMapping roots joined for each pair"),
    llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> jointTechMappingLimit(
    "joint-tech-mapping-limit",
    llvm::cl::desc("maximum TechMapping candidates admitted to Spatial PnR "
                   "for each target Module"),
    llvm::cl::init(0));
llvm::cl::opt<std::string> resolvedConfigOutputPath(
    "resolved-config-output",
    llvm::cl::desc("optional canonical executed ResolvedConfig JSON output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::uint64_t>
    workerCount("workers", llvm::cl::desc("concurrent plan workers"),
                llvm::cl::init(1));
llvm::cl::opt<std::uint64_t> siteCpu("site-cpu",
                                     llvm::cl::desc("site CPU-core capacity"),
                                     llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    siteMemory("site-memory-bytes",
               llvm::cl::desc("site memory capacity in bytes"),
               llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    siteScratch("site-scratch-bytes",
                llvm::cl::desc("site scratch capacity in bytes"),
                llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    workCpu("work-cpu", llvm::cl::desc("default in-process CPU claim"),
            llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    workMemory("work-memory-bytes",
               llvm::cl::desc("default in-process memory claim in bytes"),
               llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    workScratch("work-scratch-bytes",
                llvm::cl::desc("default in-process scratch claim in bytes"),
                llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> maximumDispatches(
    "maximum-dispatches",
    llvm::cl::desc(
        "stop after this many new work-unit dispatches; zero means unlimited"),
    llvm::cl::init(0));

llvm::cl::opt<std::string> localToolConfigPath(
    "local-tool-config",
    llvm::cl::desc("local external-tool execution configuration"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<bool> prepareOnly(
    "prepare-only",
    llvm::cl::desc("prepare external invocations without executing them"),
    llvm::cl::init(false));
llvm::cl::opt<std::uint64_t>
    externalCpu("external-cpu",
                llvm::cl::desc("CPU claim for external execution"),
                llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    externalMemory("external-memory-bytes",
                   llvm::cl::desc("memory claim for external execution"),
                   llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    externalScratch("external-scratch-bytes",
                    llvm::cl::desc("scratch claim for external execution"),
                    llvm::cl::init(0));
llvm::cl::opt<bool> claimLicense(
    "claim-license",
    llvm::cl::desc("claim one exact license unit per external binding"),
    llvm::cl::init(false));
llvm::cl::list<std::string> externalBindingCapacities(
    "external-binding-capacity",
    llvm::cl::desc("exact external binding digest and capacity as HEX=UNITS"),
    llvm::cl::value_desc("binding"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> licenseBindingCapacities(
    "license-binding-capacity",
    llvm::cl::desc("exact license binding digest and capacity as HEX=UNITS"),
    llvm::cl::value_desc("binding"), llvm::cl::ZeroOrMore);

enum class GroundTruthCampaignKind : std::uint8_t {
  None,
  Generic,
  Fpa,
};

llvm::cl::opt<GroundTruthCampaignKind> groundTruthCampaign(
    "ground-truth-campaign",
    llvm::cl::desc("apply generic or FPA campaign time gates"),
    llvm::cl::values(clEnumValN(GroundTruthCampaignKind::Generic, "generic",
                                "generic ground-truth collection policy"),
                     clEnumValN(GroundTruthCampaignKind::Fpa, "fpa",
                                "four-hour FPA active-time policy")),
    llvm::cl::init(GroundTruthCampaignKind::None));
llvm::cl::opt<std::uint64_t>
    pilotDispatchCount("pilot-dispatches",
                       llvm::cl::desc("deterministic pilot prefix size"),
                       llvm::cl::init(1));
llvm::cl::opt<std::uint64_t> minimumPilotObservations(
    "minimum-pilot-observations",
    llvm::cl::desc("minimum terminal pilot work units for admission"),
    llvm::cl::init(1));
llvm::cl::opt<std::string> progressPath(
    "progress-jsonl",
    llvm::cl::desc("append removable operational projections to path or '-'"),
    llvm::cl::value_desc("path"), llvm::cl::init("-"));
llvm::cl::opt<std::uint64_t> progressIntervalMilliseconds(
    "progress-interval-ms",
    llvm::cl::desc("live projection interval in milliseconds"),
    llvm::cl::init(1000));

volatile std::sig_atomic_t stopSignal = 0;

void requestStopFromSignal(int) { stopSignal = 1; }

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "loom_dse_invalid: " + message);
}

llvm::Expected<ArtifactRootReference> loadRootReference(llvm::StringRef path) {
  return loadArtifactRootReferenceJsonFile(path);
}

llvm::Expected<std::vector<ArtifactRootReference>>
loadRootReferences(llvm::ArrayRef<std::string> paths) {
  std::vector<ArtifactRootReference> roots;
  roots.reserve(paths.size());
  for (const std::string &path : paths) {
    auto root = loadRootReference(path);
    if (!root)
      return llvm::joinErrors(
          invalid(llvm::Twine("cannot load root binding '") + path + "'"),
          root.takeError());
    roots.push_back(std::move(*root));
  }
  llvm::sort(roots, artifactRootReferenceLess);
  if (std::adjacent_find(roots.begin(), roots.end()) != roots.end())
    return invalid("root bindings contain a duplicate reference");
  return roots;
}

llvm::Expected<std::vector<ArtifactRootReference>>
publishNormalizedPhysicalTimingProfiles(
    llvm::ArrayRef<ArtifactRootReference> systems, const ArtifactStore &store) {
  std::map<ArtifactIdentity::Storage, ArtifactRootReference> profiles;
  for (const ArtifactRootReference &systemReference : systems) {
    auto artifact =
        loom::fabric::importEntireFabricRoot(systemReference, store);
    if (!artifact)
      return artifact.takeError();
    auto system = loom::fabric::requireSystemRoot(artifact->view());
    if (!system)
      return system.takeError();
    for (const loom::fabric::AccCoreOccurrenceRef core :
         system->artifact().accCoreOccurrences()) {
      const auto target = system->spatialCoreTarget(core);
      if (!target || target->dependencyOrdinal >=
                         system->artifact().importedModules().size())
        return invalid("joint System AccCore target does not resolve");
      const auto &module =
          system->artifact().importedModules()[target->dependencyOrdinal];
      if (profiles.count(module.identity().bytes()))
        continue;
      auto profile =
          loom::fabric::projectNormalizedFabricPhysicalTimingProfile(module);
      if (!profile)
        return profile.takeError();
      auto published =
          loom::fabric::publishFabricPhysicalTimingProfile(*profile, store);
      if (!published)
        return published.takeError();
      profiles.emplace(module.identity().bytes(), std::move(*published));
    }
  }
  std::vector<ArtifactRootReference> result;
  result.reserve(profiles.size());
  for (auto &[identity, reference] : profiles) {
    (void)identity;
    result.push_back(std::move(reference));
  }
  llvm::sort(result, artifactRootReferenceLess);
  return result;
}

void canonicalizeRootUnion(std::vector<ArtifactRootReference> &roots) {
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

llvm::Error writeResolvedConfig(llvm::StringRef path,
                                const ResolvedConfig &config) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::errorCodeToError(error);
  output << canonicalResolvedConfigJson(config) << '\n';
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> parsePositiveInteger(llvm::StringRef text) {
  std::uint64_t value = 0;
  if (text.getAsInteger(10, value) || value == 0)
    return invalid("resource capacity must be a positive integer");
  return value;
}

llvm::Expected<std::vector<CountedSiteResource>>
parseCapacities(llvm::ArrayRef<std::string> spellings, bool license) {
  std::vector<CountedSiteResource> capacities;
  capacities.reserve(spellings.size());
  for (const std::string &spelling : spellings) {
    const std::size_t separator = spelling.rfind('=');
    if (separator == std::string::npos)
      return invalid("binding capacity must use HEX=UNITS");
    auto digest =
        parseBlobDigestHex(llvm::StringRef(spelling).take_front(separator));
    if (!digest)
      return digest.takeError();
    auto units = parsePositiveInteger(
        llvm::StringRef(spelling).drop_front(separator + 1));
    if (!units)
      return units.takeError();
    capacities.push_back({license
                              ? SiteResourceKey::licenseBinding(*digest)
                              : SiteResourceKey::externalToolBinding(*digest),
                          *units});
  }
  llvm::sort(capacities,
             [](const CountedSiteResource &lhs,
                const CountedSiteResource &rhs) { return lhs.key < rhs.key; });
  for (std::size_t index = 1; index < capacities.size(); ++index)
    if (capacities[index - 1].key == capacities[index].key)
      return invalid("binding capacities contain a duplicate digest");
  return capacities;
}

llvm::StringRef admissionReason(CampaignAdmissionFailureReason reason) {
  switch (reason) {
  case CampaignAdmissionFailureReason::InsufficientPilotObservations:
    return "insufficient_pilot_observations";
  case CampaignAdmissionFailureReason::PreparedAttemptIncomplete:
    return "prepared_attempt_incomplete";
  case CampaignAdmissionFailureReason::SampleActiveWallTimeLimit:
    return "sample_active_wall_time_limit";
  case CampaignAdmissionFailureReason::CampaignActiveWallTimeLimit:
    return "campaign_active_wall_time_limit";
  case CampaignAdmissionFailureReason::EstimatedCompletionLimit:
    return "estimated_completion_limit";
  case CampaignAdmissionFailureReason::ThroughputUnavailable:
    return "throughput_unavailable";
  }
  llvm_unreachable("closed campaign admission reason");
}

int reportPlanOutcome(const DsePlanExecutionOutcome &outcome) {
  if (std::holds_alternative<CompletedDsePlanExecution>(outcome)) {
    llvm::errs() << "campaign_result=completed search_complete=true\n";
    return EXIT_SUCCESS;
  }
  const auto &incomplete = std::get<IncompleteDsePlanExecution>(outcome);
  if (!incomplete.executionStopped()) {
    llvm::errs() << "campaign_result=completed search_complete=false node="
                 << incomplete.nodeOrdinal()
                 << " reason=" << toString(incomplete.reason()) << '\n';
    return EXIT_SUCCESS;
  }
  llvm::errs() << "campaign_result=incomplete node=" << incomplete.nodeOrdinal()
               << " search_complete=false reason="
               << toString(incomplete.reason()) << '\n';
  return 2;
}

std::size_t outputCount(const CompletedDsePlanExecution &completed,
                        llvm::ArrayRef<PlanOutputRef> outputs) {
  std::size_t count = 0;
  for (PlanOutputRef output : outputs)
    if (completed.hasOutput(output))
      count += completed.resolve(output).size();
  return count;
}

void reportJointOutputs(const DsePlanExecutionOutcome &outcome,
                        llvm::ArrayRef<JointDesignPlanPair> pairs) {
  const CompletedDsePlanExecution *completed =
      std::get_if<CompletedDsePlanExecution>(&outcome);
  if (!completed)
    completed =
        &std::get<IncompleteDsePlanExecution>(outcome).availableExecution();
  for (std::size_t index = 0; index != pairs.size(); ++index) {
    const JointDesignPlanPair &pair = pairs[index];
    llvm::errs() << "joint_pair=" << index << " tech_mappings="
                 << outputCount(*completed, pair.techMappings)
                 << " spatial_mappings="
                 << outputCount(*completed, pair.spatialMappings)
                 << " system_mappings="
                 << (completed->hasOutput(pair.systemMappings)
                         ? completed->resolve(pair.systemMappings).size()
                         : 0)
                 << '\n';
  }
}

llvm::Expected<int> run() {
  loom::fabric::FabricArtifactImportSession fabricImportSession;
  llvm::scope_exit emitFabricImportStatistics([&] {
    loom::fabric::emitFabricArtifactImportSessionStatistics(
        loom::fabric::FabricArtifactImportVerificationDomain::SourceInvocation,
        loom::InvocationDiagnosticStage::SpatialPnr,
        fabricImportSession.statistics());
  });
  if (progressIntervalMilliseconds == 0)
    return invalid("progress interval must be positive");
  if (prepareOnly && localToolConfigPath.empty())
    return invalid("prepare-only requires a local tool configuration");
  const bool authorJointPlan =
      !jointApplicationScopeFiles.empty() || !jointSystemRootFiles.empty();
  if (jointApplicationScopeFiles.empty() != jointSystemRootFiles.empty())
    return invalid("joint plan authoring requires both application scopes "
                   "and System root frontiers");
  if (authorJointPlan && jointSpatialMappingLimit == 0)
    return invalid("joint plan authoring requires a positive SpatialMapping "
                   "join limit");
  if (!authorJointPlan && (jointPairLimit.getNumOccurrences() != 0 ||
                           jointSpatialMappingLimit.getNumOccurrences() != 0))
    return invalid("joint policy requires joint software and System roots");
  if (!llvm::sys::fs::is_directory(artifactStorePath) ||
      !llvm::sys::fs::is_directory(blobStorePath))
    return invalid("ArtifactStore and BlobStore roots must already exist");
  if (std::error_code error = llvm::sys::fs::create_directories(runRoot))
    return llvm::errorCodeToError(error);

  if (llvm::Error error = registerProductionDseOwners())
    return error;
  auto config = loadResolvedConfig(configPath);
  if (!config)
    return config.takeError();
  auto semanticInputs = loadRootReferences(semanticInputFiles);
  if (!semanticInputs)
    return semanticInputs.takeError();
  auto preexistingEvidence = loadRootReferences(evidenceInputFiles);
  if (!preexistingEvidence)
    return preexistingEvidence.takeError();

  ArtifactStore artifacts(artifactStorePath);
  BlobStore blobs(blobStorePath);
  std::vector<JointDesignPlanPair> jointPairOutputs;
  if (authorJointPlan) {
    std::vector<std::vector<ArtifactRootReference>> applicationScopes;
    applicationScopes.reserve(jointApplicationScopeFiles.size());
    for (const std::string &path : jointApplicationScopeFiles) {
      auto scope = loadArtifactRootReferenceSetJsonFile(path);
      if (!scope)
        return llvm::joinErrors(
            invalid(llvm::Twine("cannot load application scope '") + path +
                    "'"),
            scope.takeError());
      applicationScopes.push_back(std::move(*scope));
    }
    auto systems = loadRootReferences(jointSystemRootFiles);
    if (!systems)
      return systems.takeError();
    if (jointNormalizedPhysicalTiming &&
        !jointPhysicalTimingProfileFiles.empty())
      return invalid("joint normalized timing and exact timing profile roots "
                     "are mutually exclusive");
    if (!jointNormalizedPhysicalTiming &&
        jointPhysicalTimingProfileFiles.empty())
      return invalid("joint plan requires exact physical timing profile roots "
                     "or --joint-normalized-physical-timing");
    llvm::Expected<std::vector<ArtifactRootReference>> timingProfiles =
        jointNormalizedPhysicalTiming
            ? publishNormalizedPhysicalTimingProfiles(*systems, artifacts)
            : loadRootReferences(jointPhysicalTimingProfileFiles);
    if (!timingProfiles)
      return timingProfiles.takeError();
    if (applicationScopes.size() >
        std::numeric_limits<std::uint64_t>::max() / systems->size())
      return invalid("joint pair count overflows u64");
    const std::uint64_t completePairCount =
        static_cast<std::uint64_t>(applicationScopes.size()) *
        static_cast<std::uint64_t>(systems->size());
    const std::uint64_t pairLimit =
        jointPairLimit == 0 ? completePairCount : jointPairLimit;
    auto policy =
        JointDesignPolicy::get(applicationScopes.size(), systems->size(),
                               pairLimit, jointTechMappingLimit,
                               jointSpatialMappingLimit);
    if (!policy)
      return policy.takeError();
    auto plan = buildJointDesignExplorationPlan(
        {std::move(applicationScopes), std::move(*systems)}, *timingProfiles,
        *policy, *config, artifacts);
    if (!plan)
      return plan.takeError();
    std::vector<ArtifactRootReference> jointInputs =
        projectJointDesignSemanticInputs(*plan);
    semanticInputs->insert(semanticInputs->end(), jointInputs.begin(),
                           jointInputs.end());
    canonicalizeRootUnion(*semanticInputs);
    llvm::errs() << "joint_frontier_eligible="
                 << plan->frontier.eligiblePairCount
                 << " retained=" << plan->frontier.pairs.size()
                 << " truncated=" << (plan->frontier.truncated ? 1 : 0)
                 << " analytic_evaluated="
                 << plan->frontier.analyticEvaluatedPairCount
                 << " analytic_deferred="
                 << plan->frontier.analyticDeferredPairCount << '\n';
    jointPairOutputs = plan->pairOutputs;
    *config = std::move(plan->resolvedConfig);
  }
  auto view = projectResolvedDseConfigView(*config);
  if (!view)
    return view.takeError();
  if (!resolvedConfigOutputPath.empty())
    if (llvm::Error error =
            writeResolvedConfig(resolvedConfigOutputPath, *config))
      return std::move(error);
  auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                       canonicalResolvedConfigBytes(*config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(*config))
    return invalid("ResolvedConfig publication changed its identity");
  auto producer = DseProducerSemanticBuildIdentity::get(producerBuild);
  if (!producer)
    return producer.takeError();
  auto closure = DseRunClosure::get(std::move(*producer), *semanticInputs,
                                    *config, *preexistingEvidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto journal = openExecutionJournal(runRoot, *closure, *view);
  if (!journal)
    return journal.takeError();

  auto toolCapacities = parseCapacities(externalBindingCapacities, false);
  if (!toolCapacities)
    return toolCapacities.takeError();
  auto licenseCapacities = parseCapacities(licenseBindingCapacities, true);
  if (!licenseCapacities)
    return licenseCapacities.takeError();
  auto capacity = SiteCapacity::get(siteCpu, siteMemory, siteScratch,
                                    *toolCapacities, *licenseCapacities);
  if (!capacity)
    return capacity.takeError();
  auto scheduler = SiteScheduler::create(std::move(*capacity));
  if (!scheduler)
    return scheduler.takeError();
  auto inProcessClaim =
      SiteResourceClaim::get(workCpu, workMemory, workScratch);
  if (!inProcessClaim)
    return inProcessClaim.takeError();

  std::optional<ExternalExecutionSite> externalSite;
  if (!localToolConfigPath.empty()) {
    auto localConfig = external_tool::loadLocalToolConfig(localToolConfigPath);
    if (!localConfig)
      return localConfig.takeError();
    externalSite = ExternalExecutionSite{
        std::move(*localConfig),
        prepareOnly ? ExternalAttemptDisposition::PrepareOnly
                    : ExternalAttemptDisposition::ExecutePrepared,
        externalCpu,
        externalMemory,
        externalScratch,
        claimLicense};
  }
  auto executionPolicy = PlanExecutionPolicy::get(
      workerCount, std::move(*inProcessClaim), std::move(externalSite), {},
      maximumDispatches == 0 ? std::optional<std::uint64_t>{}
                             : std::optional<std::uint64_t>{maximumDispatches});
  if (!executionPolicy)
    return executionPolicy.takeError();

  std::unique_ptr<llvm::raw_fd_ostream> progressFile;
  llvm::raw_ostream *progress = &llvm::outs();
  if (progressPath != "-") {
    std::error_code error;
    progressFile = std::make_unique<llvm::raw_fd_ostream>(
        progressPath, error, llvm::sys::fs::OF_Append);
    if (error)
      return llvm::errorCodeToError(error);
    progress = progressFile.get();
  }

  stopSignal = 0;
  std::signal(SIGINT, requestStopFromSignal);
  std::signal(SIGTERM, requestStopFromSignal);
  std::atomic<bool> monitoring{true};
  std::condition_variable monitorChanged;
  std::mutex monitorMutex;
  std::string monitorError;
  std::thread monitor([&] {
    std::unique_lock<std::mutex> lock(monitorMutex);
    while (monitoring.load(std::memory_order_relaxed)) {
      lock.unlock();
      if (stopSignal != 0 && !journal->gracefulStopRequested()) {
        if (llvm::Error error = stopDseExecution(
                *journal, GracefulStopPolicy::FinishAtomicOwnerBoundary)) {
          monitorError = llvm::toString(std::move(error));
          break;
        }
      }
      auto projection =
          projectDseOperationalState(*journal, *scheduler, workerCount);
      if (!projection) {
        monitorError = llvm::toString(projection.takeError());
        break;
      }
      if (llvm::Error error =
              writeDseOperationalProjectionJsonLine(*projection, *progress)) {
        monitorError = llvm::toString(std::move(error));
        break;
      }
      progress->flush();
      lock.lock();
      monitorChanged.wait_for(
          lock, std::chrono::milliseconds(progressIntervalMilliseconds),
          [&] { return !monitoring.load(std::memory_order_relaxed); });
    }
  });

  using ExecutionResult =
      std::variant<DsePlanExecutionOutcome, CampaignExecutionResult>;
  llvm::Expected<ExecutionResult> executionResult =
      [&]() -> llvm::Expected<ExecutionResult> {
    const GroundTruthCampaignKind campaignKind = groundTruthCampaign.getValue();
    if (campaignKind == GroundTruthCampaignKind::None) {
      auto outcome = resumeDsePlan(*view, *closure, *journal, *scheduler,
                                   *executionPolicy, artifacts, blobs);
      if (!outcome)
        return outcome.takeError();
      return ExecutionResult{std::in_place_index<0>, std::move(*outcome)};
    }
    if (campaignKind == GroundTruthCampaignKind::Generic) {
      auto campaignPolicy = CampaignExecutionPolicy::get(
          pilotDispatchCount, minimumPilotObservations);
      if (!campaignPolicy)
        return campaignPolicy.takeError();
      auto outcome = runGroundTruthCampaign(*view, *closure, *campaignPolicy,
                                            *executionPolicy, *scheduler,
                                            *journal, artifacts, blobs);
      if (!outcome)
        return outcome.takeError();
      return ExecutionResult{std::in_place_index<1>, std::move(*outcome)};
    }
    auto campaignPolicy = makeFpaGroundTruthCampaignPolicy(
        pilotDispatchCount, minimumPilotObservations);
    if (!campaignPolicy)
      return campaignPolicy.takeError();
    auto outcome = runFpaGroundTruthCampaign(*view, *closure, *campaignPolicy,
                                             *executionPolicy, *scheduler,
                                             *journal, artifacts, blobs);
    if (!outcome)
      return outcome.takeError();
    return ExecutionResult{std::in_place_index<1>, std::move(*outcome)};
  }();
  monitoring.store(false, std::memory_order_relaxed);
  monitorChanged.notify_all();
  monitor.join();
  if (!executionResult) {
    llvm::Error executionError = executionResult.takeError();
    if (!monitorError.empty())
      return llvm::joinErrors(std::move(executionError), invalid(monitorError));
    return std::move(executionError);
  }
  if (!monitorError.empty())
    return invalid(monitorError);

  auto finalProjection =
      projectDseOperationalState(*journal, *scheduler, workerCount);
  if (!finalProjection)
    return finalProjection.takeError();
  if (llvm::Error error =
          writeDseOperationalProjectionJsonLine(*finalProjection, *progress))
    return error;
  progress->flush();

  int exitCode = EXIT_SUCCESS;
  if (groundTruthCampaign.getValue() != GroundTruthCampaignKind::None) {
    CampaignExecutionResult &campaignOutcome =
        std::get<CampaignExecutionResult>(*executionResult);
    if (const auto *refused =
            std::get_if<CampaignAdmissionRefusal>(&campaignOutcome)) {
      llvm::errs() << "campaign_result=admission_refused reason="
                   << admissionReason(refused->reason) << '\n';
      exitCode = 3;
    } else {
      exitCode = reportPlanOutcome(
          std::get<CampaignExecution>(campaignOutcome).outcome);
    }
  } else {
    DsePlanExecutionOutcome &outcome =
        std::get<DsePlanExecutionOutcome>(*executionResult);
    if (!jointPairOutputs.empty())
      reportJointOutputs(outcome, jointPairOutputs);
    exitCode = reportPlanOutcome(outcome);
  }
  return exitCode;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "execute one resolved Loom DSE plan\n");
  auto result = run();
  if (!result) {
    llvm::errs() << "error: " << llvm::toString(result.takeError()) << '\n';
    return EXIT_FAILURE;
  }
  return *result;
}
