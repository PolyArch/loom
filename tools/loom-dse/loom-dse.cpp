#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CampaignRunner.h"
#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/ModelParameterCalibrationAcquisition.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"
#include "DSE/PortableSystemRtlCandidateGenerator.h"
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
#include "Evaluation/CaseText.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
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

llvm::cl::opt<bool> groundTruthCampaign(
    "ground-truth-campaign",
    llvm::cl::desc("apply deterministic pilot and collection time gates"),
    llvm::cl::init(false));
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

llvm::Error registerProductionOwners() {
  if (llvm::Error error = evaluation::registerProductionEvaluationRegistry())
    return error;
  const std::array registrations = {
      &registerStructuredOwnershipCandidateGenerator,
      &registerStructuredScheduleCandidateGenerator,
      &registerStructuredExecutionShapeCandidateGenerator,
      &registerStructuredSpecialMathAccuracyCandidateGenerator,
      &registerStructuredMemoryCommunicationCandidateGenerator,
      &registerDataflowRewriteCandidateGenerator,
      &registerSpatialPnrCandidateGenerator,
      &registerRootCompleteTechMappingCandidateGenerator,
      &registerRootCompleteSpatialPnrCandidateGenerator,
      &registerSpatialMappingFeedbackCandidateGenerator,
      &registerRootCompleteSystemPnrCandidateGenerator,
      &registerFabricTemplateCandidateGenerator,
      &registerSpatialTopologyCandidateGenerator,
      &registerSpatialMicroarchitectureCandidateGenerator,
      &registerSystemCompositionCandidateGenerator,
      &registerPortableSystemRtlCandidateGenerator,
      &registerFpaGbdtTrainingCandidateGenerator,
      &registerSystemRuntimeGbdtTrainingCandidateGenerator,
      &registerStructuredEvaluationPromotionAcquisition,
      &registerDataflowEvaluationPromotionAcquisition,
      &registerSpatialMappingEvaluationPromotionAcquisition,
      &registerModelParameterCalibrationPromotionAcquisitions};
  for (llvm::Error (*registration)() : registrations)
    if (llvm::Error error = registration())
      return error;
  return llvm::Error::success();
}

llvm::Expected<ArtifactRootReference> loadRootReference(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::errorCodeToError(buffer.getError());
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("root binding file must contain one JSON object");
  return evaluation::parseArtifactRootReferenceJson(*object);
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
    llvm::errs() << "campaign_result=completed\n";
    return EXIT_SUCCESS;
  }
  const auto &incomplete = std::get<IncompleteDsePlanExecution>(outcome);
  llvm::errs() << "campaign_result=incomplete node=" << incomplete.nodeOrdinal()
               << " reason=" << toString(incomplete.reason()) << '\n';
  return 2;
}

llvm::Expected<int> run() {
  if (progressIntervalMilliseconds == 0)
    return invalid("progress interval must be positive");
  if (prepareOnly && localToolConfigPath.empty())
    return invalid("prepare-only requires a local tool configuration");
  if (!llvm::sys::fs::is_directory(artifactStorePath) ||
      !llvm::sys::fs::is_directory(blobStorePath))
    return invalid("ArtifactStore and BlobStore roots must already exist");
  if (std::error_code error = llvm::sys::fs::create_directories(runRoot))
    return llvm::errorCodeToError(error);

  if (llvm::Error error = registerProductionOwners())
    return error;
  auto config = loadResolvedConfig(configPath);
  if (!config)
    return config.takeError();
  auto view = projectResolvedDseConfigView(*config);
  if (!view)
    return view.takeError();
  auto semanticInputs = loadRootReferences(semanticInputFiles);
  if (!semanticInputs)
    return semanticInputs.takeError();
  auto preexistingEvidence = loadRootReferences(evidenceInputFiles);
  if (!preexistingEvidence)
    return preexistingEvidence.takeError();

  ArtifactStore artifacts(artifactStorePath);
  BlobStore blobs(blobStorePath);
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
    if (!groundTruthCampaign) {
      auto outcome = resumeDsePlan(*view, *closure, *journal, *scheduler,
                                   *executionPolicy, artifacts, blobs);
      if (!outcome)
        return outcome.takeError();
      return ExecutionResult{std::in_place_index<0>, std::move(*outcome)};
    }
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
  }();
  monitoring.store(false, std::memory_order_relaxed);
  monitorChanged.notify_all();
  monitor.join();
  if (!monitorError.empty())
    return invalid(monitorError);
  if (!executionResult)
    return executionResult.takeError();

  auto finalProjection =
      projectDseOperationalState(*journal, *scheduler, workerCount);
  if (!finalProjection)
    return finalProjection.takeError();
  if (llvm::Error error =
          writeDseOperationalProjectionJsonLine(*finalProjection, *progress))
    return error;
  progress->flush();

  int exitCode = EXIT_SUCCESS;
  if (groundTruthCampaign) {
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
    exitCode =
        reportPlanOutcome(std::get<DsePlanExecutionOutcome>(*executionResult));
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
