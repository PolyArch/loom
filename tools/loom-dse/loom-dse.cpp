#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CampaignRunner.h"
#include "DSE/GroundTruthPlan.h"
#include "DSE/HardwareImplementationEvaluationAcquisition.h"
#include "EDA/Adapters/OpenSource/OpenRoadStaticFpa.h"
#include "Evaluation/CaseText.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

enum class Command { Run, Prepare, Status };

llvm::cl::opt<Command> command(
    llvm::cl::Positional, llvm::cl::desc("<command>"), llvm::cl::Required,
    llvm::cl::values(clEnumValN(Command::Run, "run",
                                "run or resume the campaign"),
                     clEnumValN(Command::Prepare, "prepare",
                                "prepare the deterministic campaign prefix"),
                     clEnumValN(Command::Status, "status",
                                "project the current operational state")));

llvm::cl::opt<std::string>
    resolvedConfigPath("resolved-config",
                       llvm::cl::desc("canonical ResolvedConfig JSON path"),
                       llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    bindingsPath("bindings", llvm::cl::desc("run input bindings JSON path"),
                 llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("ArtifactStore directory"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string> blobStorePath("blob-store",
                                         llvm::cl::desc("BlobStore directory"),
                                         llvm::cl::value_desc("path"),
                                         llvm::cl::Required);
llvm::cl::opt<std::string> runRoot("run-root",
                                   llvm::cl::desc("campaign journal directory"),
                                   llvm::cl::value_desc("path"),
                                   llvm::cl::Required);
llvm::cl::opt<std::string>
    producerBuild("producer-build",
                  llvm::cl::desc("stable producer semantic/build identity"),
                  llvm::cl::value_desc("identity"), llvm::cl::Required);
llvm::cl::opt<std::string>
    localToolConfigPath("local-tool-config",
                        llvm::cl::desc("machine-local tool config path"),
                        llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> outputPath("o", llvm::cl::desc("JSONL output path"),
                                      llvm::cl::value_desc("path"),
                                      llvm::cl::init("-"));

llvm::cl::opt<std::uint64_t>
    workers("workers",
            llvm::cl::desc("maximum concurrently dispatched work units"),
            llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    siteCpuCores("site-cpu-cores", llvm::cl::desc("declared site CPU capacity"),
                 llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    siteMemoryBytes("site-memory-bytes",
                    llvm::cl::desc("declared site memory capacity"),
                    llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    siteScratchBytes("site-scratch-bytes",
                     llvm::cl::desc("declared site scratch capacity"),
                     llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    externalCpuCores("external-cpu-cores",
                     llvm::cl::desc("CPU claim per external attempt"),
                     llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    externalMemoryBytes("external-memory-bytes",
                        llvm::cl::desc("memory claim per external attempt"),
                        llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    externalScratchBytes("external-scratch-bytes",
                         llvm::cl::desc("scratch claim per external attempt"),
                         llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> externalToolSlots(
    "external-tool-slots",
    llvm::cl::desc("capacity for each admitted execution binding"),
    llvm::cl::init(1));
llvm::cl::opt<bool> claimLicense(
    "claim-license",
    llvm::cl::desc("claim one license per external execution binding"),
    llvm::cl::init(false));
llvm::cl::list<std::string> externalBindingDigests(
    "external-binding-digest",
    llvm::cl::desc("admit an external execution binding digest"),
    llvm::cl::value_desc("sha256"), llvm::cl::ZeroOrMore);

llvm::cl::opt<std::uint64_t>
    pilotDispatches("pilot-dispatches",
                    llvm::cl::desc("deterministic prefix dispatch bound"),
                    llvm::cl::init(4));
llvm::cl::opt<std::uint64_t> minimumPilotObservations(
    "minimum-pilot-observations",
    llvm::cl::desc("minimum terminal work units required after the pilot"),
    llvm::cl::init(1));
llvm::cl::opt<std::uint64_t> maximumDispatches(
    "maximum-dispatches",
    llvm::cl::desc("optional whole-run dispatch bound; zero is unbounded"),
    llvm::cl::init(0));

volatile std::sig_atomic_t interrupted = 0;

void interruptHandler(int) { interrupted = 1; }

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "loom_dse_invalid: " + message);
}

int reportError(llvm::Error error) {
  llvm::errs() << "loom-dse: " << llvm::toString(std::move(error)) << '\n';
  return 1;
}

struct RunBindings final {
  std::vector<loom::ArtifactRootReference> semanticInputs;
  std::vector<loom::ArtifactRootReference> preexistingEvidence;
};

llvm::Expected<std::vector<loom::ArtifactRootReference>>
parseReferences(const llvm::json::Object &object, llvm::StringRef field,
                bool required) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array) {
    if (!required && !object.get(field))
      return std::vector<loom::ArtifactRootReference>{};
    return invalid("bindings field '" + field + "' must be an array");
  }
  std::vector<loom::ArtifactRootReference> references;
  references.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    const llvm::json::Object *reference = value.getAsObject();
    if (!reference)
      return invalid("bindings field '" + field +
                     "' contains a non-object reference");
    auto parsed = loom::evaluation::parseArtifactRootReferenceJson(*reference);
    if (!parsed)
      return parsed.takeError();
    references.push_back(std::move(*parsed));
  }
  return references;
}

llvm::Expected<RunBindings> loadRunBindings(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "cannot read run bindings");
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object)
    return invalid("run bindings root must be an object");
  for (const auto &field : *object)
    if (field.first != "semantic_inputs" &&
        field.first != "preexisting_evidence")
      return invalid("unknown run bindings field '" + field.first.str() + "'");
  auto semanticInputs = parseReferences(*object, "semantic_inputs", true);
  if (!semanticInputs)
    return semanticInputs.takeError();
  auto evidence = parseReferences(*object, "preexisting_evidence", false);
  if (!evidence)
    return evidence.takeError();
  return RunBindings{std::move(*semanticInputs), std::move(*evidence)};
}

llvm::Expected<std::vector<loom::BlobDigest>>
preparedBindingDigests(const loom::dse::ExecutionJournal &journal) {
  auto records = journal.workUnits();
  if (!records)
    return records.takeError();
  std::vector<loom::BlobDigest> digests;
  for (const loom::dse::JournalWorkUnitRecord &record : *records) {
    if (!record.preparedInvocation)
      continue;
    auto digest = loom::external_tool::deriveExternalToolExecutionBindingDigest(
        *record.preparedInvocation);
    if (!digest)
      return digest.takeError();
    digests.push_back(std::move(*digest));
  }
  llvm::sort(digests,
             [](const loom::BlobDigest &lhs, const loom::BlobDigest &rhs) {
               return lhs.bytes() < rhs.bytes();
             });
  digests.erase(std::unique(digests.begin(), digests.end()), digests.end());
  return digests;
}

llvm::Expected<std::vector<loom::BlobDigest>>
admittedBindingDigests(const loom::dse::ExecutionJournal &journal) {
  auto digests = preparedBindingDigests(journal);
  if (!digests)
    return digests.takeError();
  for (const std::string &spelling : externalBindingDigests) {
    auto digest = loom::parseBlobDigestHex(spelling);
    if (!digest)
      return digest.takeError();
    digests->push_back(std::move(*digest));
  }
  llvm::sort(*digests,
             [](const loom::BlobDigest &lhs, const loom::BlobDigest &rhs) {
               return lhs.bytes() < rhs.bytes();
             });
  digests->erase(std::unique(digests->begin(), digests->end()), digests->end());
  return digests;
}

llvm::Expected<loom::dse::SiteScheduler>
makeScheduler(llvm::ArrayRef<loom::BlobDigest> bindings) {
  std::vector<loom::dse::CountedSiteResource> tools;
  std::vector<loom::dse::CountedSiteResource> licenses;
  tools.reserve(bindings.size());
  licenses.reserve(claimLicense ? bindings.size() : 0);
  for (const loom::BlobDigest &binding : bindings) {
    tools.push_back({loom::dse::SiteResourceKey::externalToolBinding(binding),
                     externalToolSlots});
    if (claimLicense)
      licenses.push_back({loom::dse::SiteResourceKey::licenseBinding(binding),
                          externalToolSlots});
  }
  llvm::sort(tools, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  llvm::sort(licenses, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  auto capacity = loom::dse::SiteCapacity::get(
      siteCpuCores, siteMemoryBytes, siteScratchBytes, tools, licenses);
  if (!capacity)
    return capacity.takeError();
  return loom::dse::SiteScheduler::create(std::move(*capacity));
}

llvm::Expected<loom::dse::PlanExecutionPolicy>
makeExecutionPolicy(const loom::external_tool::LocalToolConfig &localConfig,
                    loom::dse::ExternalAttemptDisposition disposition,
                    std::optional<std::uint64_t> dispatchLimit) {
  auto inProcessClaim = loom::dse::SiteResourceClaim::get(1, 0, 0);
  if (!inProcessClaim)
    return inProcessClaim.takeError();
  loom::dse::ExternalExecutionSite externalSite{
      localConfig,         disposition,          externalCpuCores,
      externalMemoryBytes, externalScratchBytes, claimLicense};
  return loom::dse::PlanExecutionPolicy::get(
      workers, std::move(*inProcessClaim), std::move(externalSite), {},
      dispatchLimit);
}

llvm::Error writePreparedBindings(const loom::dse::ExecutionJournal &journal,
                                  llvm::raw_ostream &output) {
  auto digests = preparedBindingDigests(journal);
  if (!digests)
    return digests.takeError();
  for (const loom::BlobDigest &digest : *digests) {
    llvm::json::OStream json(output, 0);
    json.object([&] {
      json.attribute("event", "prepared_external_binding");
      json.attribute("digest", loom::formatBlobDigestHex(digest));
    });
    output << '\n';
  }
  return llvm::Error::success();
}

void writeArtifactArray(llvm::json::OStream &json,
                        llvm::ArrayRef<loom::ArtifactRootReference> artifacts) {
  json.arrayBegin();
  for (const loom::ArtifactRootReference &artifact : artifacts)
    loom::evaluation::writeArtifactRootReferenceJson(json, artifact);
  json.arrayEnd();
}

void writeCompletedOutputs(const loom::dse::ResolvedDsePlan &plan,
                           const loom::dse::CompletedDsePlanExecution &outcome,
                           llvm::raw_ostream &output) {
  for (std::size_t node = 0; node < plan.nodes().size(); ++node) {
    for (std::uint32_t slot = 0;; ++slot) {
      const loom::dse::PlanOutputRef reference{static_cast<std::uint64_t>(node),
                                               slot};
      if (!plan.resolve(reference))
        break;
      llvm::json::OStream json(output, 0);
      json.object([&] {
        json.attribute("event", "plan_output");
        json.attribute("node", static_cast<std::uint64_t>(node));
        json.attribute("slot", slot);
        json.attributeBegin("artifacts");
        writeArtifactArray(json, outcome.resolve(reference));
        json.attributeEnd();
      });
      output << '\n';
    }
  }
}

llvm::StringRef
admissionReason(loom::dse::CampaignAdmissionFailureReason reason) {
  using Reason = loom::dse::CampaignAdmissionFailureReason;
  switch (reason) {
  case Reason::InsufficientPilotObservations:
    return "insufficient_pilot_observations";
  case Reason::PreparedAttemptIncomplete:
    return "prepared_attempt_incomplete";
  case Reason::SampleActiveWallTimeLimit:
    return "sample_active_wall_time_limit";
  case Reason::CampaignActiveWallTimeLimit:
    return "campaign_active_wall_time_limit";
  case Reason::EstimatedCompletionLimit:
    return "estimated_completion_limit";
  case Reason::ThroughputUnavailable:
    return "throughput_unavailable";
  }
  llvm_unreachable("unknown CampaignAdmissionFailureReason");
}

class InterruptMonitor final {
public:
  explicit InterruptMonitor(loom::dse::ExecutionJournal &journal)
      : journal_(journal), thread_([this] { run(); }) {}
  ~InterruptMonitor() { llvm::consumeError(finish()); }

  llvm::Error finish() {
    if (!joined_) {
      done_.store(true, std::memory_order_relaxed);
      thread_.join();
      joined_ = true;
    }
    if (!stopError_)
      return llvm::Error::success();
    std::string message = std::move(*stopError_);
    stopError_.reset();
    return invalid("graceful stop failed: " + message);
  }

private:
  void run() {
    while (!done_.load(std::memory_order_relaxed)) {
      if (interrupted != 0) {
        if (llvm::Error error = loom::dse::stopDseExecution(
                journal_,
                loom::dse::GracefulStopPolicy::FinishAtomicOwnerBoundary))
          stopError_ = llvm::toString(std::move(error));
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  loom::dse::ExecutionJournal &journal_;
  std::atomic_bool done_{false};
  std::thread thread_;
  std::optional<std::string> stopError_;
  bool joined_ = false;
};

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "loom-dse: execute and resume one exact resolved DSE plan\n");
  if (llvm::Error error = loom::dse::
          registerHardwareImplementationEvaluationPromotionAcquisition())
    return reportError(std::move(error));
  if (llvm::Error error =
          loom::eda::open_source::registerOpenRoadStaticFpaEvaluationProvider())
    return reportError(std::move(error));

  auto config = loom::loadResolvedConfig(resolvedConfigPath);
  if (!config)
    return reportError(config.takeError());
  auto bindings = loadRunBindings(bindingsPath);
  if (!bindings)
    return reportError(bindings.takeError());
  loom::ArtifactStore artifacts(artifactStorePath);
  loom::BlobStore blobs(blobStorePath);
  auto configIdentity =
      artifacts.put(loom::ResolvedConfig::artifactSchema,
                    loom::canonicalResolvedConfigBytes(*config));
  if (!configIdentity)
    return reportError(configIdentity.takeError());
  if (*configIdentity != loom::resolvedConfigIdentity(*config))
    return reportError(invalid("ResolvedConfig publication changed identity"));
  auto view = loom::dse::projectResolvedDseConfigView(*config);
  if (!view)
    return reportError(view.takeError());
  auto producer =
      loom::dse::DseProducerSemanticBuildIdentity::get(producerBuild);
  if (!producer)
    return reportError(producer.takeError());
  auto closure = loom::dse::DseRunClosure::get(
      std::move(*producer), bindings->semanticInputs, *config,
      bindings->preexistingEvidence, artifacts);
  if (!closure)
    return reportError(closure.takeError());
  auto journal = loom::dse::openExecutionJournal(runRoot, *closure, *view);
  if (!journal)
    return reportError(journal.takeError());
  auto localConfig =
      localToolConfigPath.empty()
          ? llvm::Expected<loom::external_tool::LocalToolConfig>(
                loom::external_tool::defaultLocalToolConfig())
          : loom::external_tool::loadLocalToolConfig(localToolConfigPath);
  if (!localConfig)
    return reportError(localConfig.takeError());
  std::error_code outputError;
  auto output = std::make_unique<llvm::ToolOutputFile>(outputPath, outputError,
                                                       llvm::sys::fs::OF_None);
  if (outputError)
    return reportError(
        llvm::createStringError(outputError, "cannot open JSONL output"));

  auto digests = admittedBindingDigests(*journal);
  if (!digests)
    return reportError(digests.takeError());
  auto scheduler = makeScheduler(*digests);
  if (!scheduler)
    return reportError(scheduler.takeError());

  if (command == Command::Status) {
    auto projection =
        loom::dse::projectDseOperationalState(*journal, *scheduler, workers);
    if (!projection)
      return reportError(projection.takeError());
    if (llvm::Error error = loom::dse::writeDseOperationalProjectionJsonLine(
            *projection, output->os()))
      return reportError(std::move(error));
    if (llvm::Error error = writePreparedBindings(*journal, output->os()))
      return reportError(std::move(error));
    output->keep();
    return 0;
  }

  if (command == Command::Prepare) {
    auto policy = makeExecutionPolicy(
        *localConfig, loom::dse::ExternalAttemptDisposition::PrepareOnly,
        pilotDispatches);
    if (!policy)
      return reportError(policy.takeError());
    auto outcome = loom::dse::resumeDsePlan(
        *view, *closure, *journal, *scheduler, *policy, artifacts, blobs);
    if (!outcome)
      return reportError(outcome.takeError());
    auto projection =
        loom::dse::projectDseOperationalState(*journal, *scheduler, workers);
    if (!projection)
      return reportError(projection.takeError());
    if (llvm::Error error = loom::dse::writeDseOperationalProjectionJsonLine(
            *projection, output->os()))
      return reportError(std::move(error));
    if (llvm::Error error = writePreparedBindings(*journal, output->os()))
      return reportError(std::move(error));
    output->keep();
    return 0;
  }

  if (digests->empty()) {
    auto preparePolicy = makeExecutionPolicy(
        *localConfig, loom::dse::ExternalAttemptDisposition::PrepareOnly,
        pilotDispatches);
    if (!preparePolicy)
      return reportError(preparePolicy.takeError());
    auto prepared =
        loom::dse::resumeDsePlan(*view, *closure, *journal, *scheduler,
                                 *preparePolicy, artifacts, blobs);
    if (!prepared)
      return reportError(prepared.takeError());
    digests = admittedBindingDigests(*journal);
    if (!digests)
      return reportError(digests.takeError());
    scheduler = makeScheduler(*digests);
    if (!scheduler)
      return reportError(scheduler.takeError());
  }

  auto campaignPolicy = loom::dse::CampaignExecutionPolicy::get(
      pilotDispatches, minimumPilotObservations);
  if (!campaignPolicy)
    return reportError(campaignPolicy.takeError());
  auto executionPolicy = makeExecutionPolicy(
      *localConfig, loom::dse::ExternalAttemptDisposition::ExecutePrepared,
      maximumDispatches == 0 ? std::optional<std::uint64_t>{}
                             : std::optional<std::uint64_t>{maximumDispatches});
  if (!executionPolicy)
    return reportError(executionPolicy.takeError());

  std::signal(SIGINT, interruptHandler);
  InterruptMonitor monitor(*journal);
  auto result = loom::dse::runGroundTruthCampaign(
      *view, *closure, *campaignPolicy, *executionPolicy, *scheduler, *journal,
      artifacts, blobs);
  llvm::Error monitorError = monitor.finish();
  if (!result)
    return reportError(
        llvm::joinErrors(result.takeError(), std::move(monitorError)));
  if (monitorError)
    return reportError(std::move(monitorError));

  const loom::dse::DseOperationalProjection &projection = std::visit(
      [](const auto &value) -> const loom::dse::DseOperationalProjection & {
        return value.projection;
      },
      *result);
  if (llvm::Error error = loom::dse::writeDseOperationalProjectionJsonLine(
          projection, output->os()))
    return reportError(std::move(error));
  if (auto *refusal =
          std::get_if<loom::dse::CampaignAdmissionRefusal>(&*result)) {
    llvm::json::OStream json(output->os(), 0);
    json.object([&] {
      json.attribute("event", "campaign_refused");
      json.attribute("reason", admissionReason(refusal->reason));
    });
    output->os() << '\n';
    output->keep();
    return 2;
  }

  auto &execution = std::get<loom::dse::CampaignExecution>(*result);
  if (auto *completed = std::get_if<loom::dse::CompletedDsePlanExecution>(
          &execution.outcome)) {
    writeCompletedOutputs(view->plan(), *completed, output->os());
    output->keep();
    return 0;
  }
  const auto &incomplete =
      std::get<loom::dse::IncompleteDsePlanExecution>(execution.outcome);
  llvm::json::OStream json(output->os(), 0);
  json.object([&] {
    json.attribute("event", "campaign_incomplete");
    json.attribute("node", incomplete.nodeOrdinal());
    json.attribute("reason", loom::dse::toString(incomplete.reason()));
  });
  output->os() << '\n';
  output->keep();
  return 2;
}
