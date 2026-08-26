#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/ExecutionControl.h"
#include "Common/TimeoutBudgets.h"
#include "Config/ResolvedConfig.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialRuntimeFeedback.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "PnR/PnrConfig.h"
#include "Runtime/Gem5SystemExecution.h"

#include "MappedRtlSimulationTestSupport.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace {

constexpr std::uint64_t kWarmupRuns = 1;
constexpr std::uint64_t kMeasurementRuns = 3;
constexpr std::uint64_t kQualificationLimitNanoseconds = 45'000'000'000ULL;
constexpr auto kQualificationLimit = std::chrono::seconds(45);
constexpr auto kSpatialPnrQualificationLimit =
    loom::timeout::duration(loom::timeout::Tier::Fast);

class MonotonicExecutionDeadline final {
public:
  explicit MonotonicExecutionDeadline(
      std::chrono::steady_clock::duration duration)
      : notAfter_(std::chrono::steady_clock::now() + duration) {}

  loom::ExecutionControlView control() const {
    return {this, stopRequested, remainingTime};
  }

private:
  static bool stopRequested(const void *context) {
    const auto &deadline =
        *static_cast<const MonotonicExecutionDeadline *>(context);
    return std::chrono::steady_clock::now() >= deadline.notAfter_;
  }

  static std::optional<std::chrono::steady_clock::duration>
  remainingTime(const void *context) {
    const auto &deadline =
        *static_cast<const MonotonicExecutionDeadline *>(context);
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline.notAfter_)
      return std::chrono::steady_clock::duration::zero();
    return deadline.notAfter_ - now;
  }

  std::chrono::steady_clock::time_point notAfter_;
};

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "CGRA budget profile: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

llvm::json::Object referenceJson(const loom::ArtifactRootReference &reference) {
  return llvm::json::Object{
      {"schema", reference.schemaIdentity},
      {"schema_version", loom::formatSchemaVersion(reference.schemaVersion)},
      {"artifact", loom::formatArtifactIdentityHex(reference.artifact)}};
}

llvm::json::Value infeasibilityProofJson(
    const loom::dse::CandidateGeneratorProviderResult &result) {
  const auto *proven =
      std::get_if<loom::dse::ProvenInfeasibleCandidateGeneratorResult>(
          &result.outcome);
  if (!proven)
    return nullptr;
  return llvm::json::Object{
      {"kind", proven->proof.kind.ordinal()},
      {"witness", llvm::toHex(proven->proof.witness, true)}};
}

const std::vector<loom::ArtifactRootReference> &
candidateArtifacts(const loom::dse::CandidateGeneratorProviderResult &result) {
  const std::vector<loom::dse::CandidateGeneratorOutputBinding> *bindings =
      nullptr;
  if (const auto *completed =
          std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
              &result.outcome))
    bindings = &completed->outputBindings;
  else if (const auto *proven =
               std::get_if<loom::dse::ProvenInfeasibleCandidateGeneratorResult>(
                   &result.outcome))
    bindings = &proven->outputBindings;
  else
    bindings =
        &std::get<loom::dse::IncompleteCandidateGeneratorResult>(result.outcome)
             .retainedOutputBindings;
  require(bindings->size() == 1,
          "qualification generator changed its output shape");
  return bindings->front().artifacts;
}

llvm::json::Object candidateGeneratorResultJson(
    const loom::dse::CandidateGeneratorDescriptor &descriptor,
    const loom::dse::CandidateGeneratorProviderResult &result) {
  require(result.workSummary.size() == descriptor.workUnits.size(),
          "qualification generator work summary has the wrong width");
  llvm::StringRef outcome = "completed";
  std::optional<llvm::StringRef> incompleteReason;
  if (const auto *incomplete =
          std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
              &result.outcome)) {
    outcome = "incomplete";
    incompleteReason = loom::dse::candidateGeneratorIncompleteReasonSpelling(
        incomplete->reason);
  } else if (std::holds_alternative<
                 loom::dse::ProvenInfeasibleCandidateGeneratorResult>(
                 result.outcome)) {
    require(descriptor.ownerInfeasibilityProof,
            "qualification infeasibility has no descriptor proof contract");
    require(candidateArtifacts(result).empty(),
            "qualification infeasibility retained a candidate");
    outcome = "proven_infeasible";
  }

  llvm::json::Array candidates;
  for (const loom::ArtifactRootReference &candidate :
       candidateArtifacts(result))
    candidates.push_back(referenceJson(candidate));
  llvm::json::Array workUnits;
  for (const auto [ordinal, entry] : llvm::enumerate(result.workSummary)) {
    require(entry.unit.ordinal() == ordinal && entry.consumed <= entry.planned,
            "qualification generator work summary is not canonical");
    if (outcome != "incomplete")
      require(entry.planned == entry.consumed,
              "terminal qualification generator left planned work unconsumed");
    workUnits.push_back(llvm::json::Object{
        {"unit", descriptor.workUnits[ordinal].spelling.str()},
        {"planned", entry.planned},
        {"consumed", entry.consumed}});
  }
  return llvm::json::Object{
      {"outcome", outcome.str()},
      {"incomplete_reason", incompleteReason
                                ? llvm::json::Value(incompleteReason->str())
                                : llvm::json::Value(nullptr)},
      {"infeasibility_proof", infeasibilityProofJson(result)},
      {"candidates", std::move(candidates)},
      {"work_units", std::move(workUnits)}};
}

llvm::json::Object spatialPnrResultJson(
    const loom::pnr::ResolvedPnrConfigView &config,
    const loom::dse::CandidateGeneratorProviderResult &result) {
  const auto completionGoal = config.policy().search.completionGoal;
  require(completionGoal ==
              loom::ResolvedPnrCompletionGoal::ExhaustConfiguredWork,
          "qualification PnR did not select exhaustive configured work");
  require(result.workSummary.size() ==
              loom::dse::pnrCandidateGeneratorWorkUnits.size(),
          "qualification PnR work summary has the wrong width");

  llvm::StringRef outcome = "completed";
  std::optional<llvm::StringRef> incompleteReason;
  if (const auto *incomplete =
          std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
              &result.outcome)) {
    outcome = "incomplete";
    incompleteReason = loom::dse::candidateGeneratorIncompleteReasonSpelling(
        incomplete->reason);
  } else if (const auto *proven = std::get_if<
                 loom::dse::ProvenInfeasibleCandidateGeneratorResult>(
                 &result.outcome)) {
    require(proven->outputBindings.size() == 1 &&
                proven->outputBindings.front().artifacts.empty(),
            "proven-infeasible qualification PnR retained a candidate");
    outcome = "proven_infeasible";
  } else {
    const auto &completed =
        std::get<loom::dse::CompletedCandidateGeneratorResult>(result.outcome);
    require(completed.outputBindings.size() == 1,
            "qualification PnR changed its output shape");
  }
  if (outcome == "completed") {
    const std::uint64_t configuredSeedAttempts =
        config.policy().search.initializer.seedAttemptCount;
    require(result.workSummary.front().unit.ordinal() == 0 &&
                result.workSummary.front().planned >= configuredSeedAttempts,
            "qualification PnR restart plan disagrees with ResolvedConfig");
  }

  llvm::json::Array generatorSummary;
  for (const auto [ordinal, entry] : llvm::enumerate(result.workSummary)) {
    require(entry.unit.ordinal() == ordinal && entry.consumed <= entry.planned,
            "qualification PnR work summary is not canonical");
    if (outcome != "incomplete")
      require(entry.planned == entry.consumed,
              "terminal qualification PnR left planned work unconsumed");
    generatorSummary.push_back(llvm::json::Object{
        {"unit",
         loom::dse::pnrCandidateGeneratorWorkUnits[ordinal].spelling.str()},
        {"planned", entry.planned},
        {"consumed", entry.consumed}});
  }

  llvm::json::Array candidates;
  for (const loom::ArtifactRootReference &candidate :
       candidateArtifacts(result))
    candidates.push_back(referenceJson(candidate));

  return llvm::json::Object{
      {"completion_goal",
       loom::resolvedPnrCompletionGoalSpelling(completionGoal).str()},
      {"configured_seed_attempts",
       config.policy().search.initializer.seedAttemptCount},
      {"outcome", outcome.str()},
      {"incomplete_reason", incompleteReason
                                ? llvm::json::Value(incompleteReason->str())
                                : llvm::json::Value(nullptr)},
      {"infeasibility_proof", infeasibilityProofJson(result)},
      {"candidates", std::move(candidates)},
      {"work_units", std::move(generatorSummary)}};
}

struct SourceCase final {
  loom::ArtifactRootReference dataflow;
  loom::ArtifactRootReference workload;
  loom::ArtifactRootReference runtimeInput;
};

SourceCase readSourceCase(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    fail("cannot read source report: " + buffer.getError().message());
  auto value = take(llvm::json::parse((*buffer)->getBuffer()));
  const llvm::json::Object *root = value.getAsObject();
  const llvm::json::Object *artifacts =
      root ? root->getObject("artifacts") : nullptr;
  const auto dataflowSpelling =
      artifacts ? artifacts->getString("canonical_dataflow") : std::nullopt;
  const llvm::json::Array *replayCases =
      root ? root->getArray("replay_cases") : nullptr;
  const auto replayCaseOccurrences =
      root ? root->getInteger("replay_case_occurrences") : std::nullopt;
  require(dataflowSpelling.has_value(),
          "source report has no canonical Dataflow identity");
  require(replayCases && replayCases->size() == 1,
          "source report must contain one exact replay case");
  require(replayCaseOccurrences && *replayCaseOccurrences == 1,
          "source report must contain one replay occurrence");
  const llvm::json::Object *replay = replayCases->front().getAsObject();
  const llvm::json::Object *workload =
      replay ? replay->getObject("workload") : nullptr;
  const llvm::json::Object *runtimeInput =
      replay ? replay->getObject("runtime_input") : nullptr;
  require(workload && runtimeInput,
          "source report replay case is not a reference pair");
  return {{dataflow::canonicalDataflowSchema.identity.str(),
           dataflow::canonicalDataflowSchema.version,
           take(loom::parseArtifactIdentityHex(*dataflowSpelling))},
          take(loom::parseArtifactRootReferenceJson(*workload)),
          take(loom::parseArtifactRootReferenceJson(*runtimeInput))};
}

void emitClosedWaitDiagnostic(
    const loom::sim::CgraClosedWaitSetDiagnostic &diagnostic) {
  llvm::errs() << "CGRA closed wait: actors=" << diagnostic.pendingActorFirings
               << " transfers=" << diagnostic.pendingTransfers
               << " physical_actions=" << diagnostic.pendingPhysicalActions
               << " graph_retirement=" << diagnostic.graphRetirementVisible
               << '\n';
  for (const auto &edge : diagnostic.actorWaitCycle)
    llvm::errs() << "CGRA actor wait edge: waiting=" << edge.waitingActorOrdinal
                 << " blocking=" << edge.blockingActorOrdinal
                 << " kind=" << static_cast<unsigned>(edge.kind) << '\n';
  for (const auto &edge : diagnostic.transferWaitCycle)
    llvm::errs() << "CGRA transfer wait edge: waiting_binding="
                 << edge.waitingBindingOrdinal
                 << " waiting_occurrence=" << edge.waitingOccurrenceOrdinal
                 << " blocking_actor=" << edge.blockingActorOrdinal
                 << " blocking_binding=" << edge.blockingBindingOrdinal
                 << " blocking_occurrence=" << edge.blockingOccurrenceOrdinal
                 << " kind=" << static_cast<unsigned>(edge.kind) << '\n';
  for (const auto &transfer : diagnostic.transfers) {
    if (!transfer.blocked || !transfer.blockingFifoOccurrence)
      continue;
    llvm::errs() << "CGRA blocking FIFO: binding=" << transfer.bindingOrdinal
                 << " occurrence=" << transfer.occurrenceOrdinal
                 << " occupancy=" << transfer.blockingStorageOccupancy
                 << " reservations=" << transfer.blockingStorageReservations
                 << " capacity=" << transfer.blockingStorageCapacity
                 << " fifo=";
    loom::fabric::printFabricRef(llvm::errs(),
                                 *transfer.blockingFifoOccurrence);
    llvm::errs() << '\n';
  }
}

std::uint64_t referenceCycles(
    const loom::evaluation::models::CgraSimulationEvaluation &evaluation) {
  const auto &evidence = evaluation.evidence;
  const auto *completed =
      std::get_if<loom::evaluation::CompletedEvidence>(&evidence.outcome());
  if (!completed) {
    llvm::errs() << "CGRA evidence outcome: "
                 << loom::evaluation::toString(evidence.outcomeKind()) << '\n';
    if (evaluation.closedWait)
      emitClosedWaitDiagnostic(*evaluation.closedWait);
  }
  require(completed && completed->metricResults.size() == 1,
          "CGRA execution did not publish one completed metric");
  const auto *point = std::get_if<loom::evaluation::PointObservation>(
      &completed->metricResults.front().observation);
  const auto *integer =
      point ? std::get_if<loom::evaluation::IntegerValue>(&point->value)
            : nullptr;
  require(integer && integer->value() > 0,
          "CGRA execution did not publish a positive cycle count");
  return static_cast<std::uint64_t>(integer->value());
}

bool completed(
    const loom::evaluation::models::CgraSimulationEvaluation &evaluation) {
  return std::holds_alternative<loom::evaluation::CompletedEvidence>(
      evaluation.evidence.outcome());
}

std::uint64_t peakResidentBytes() {
  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0,
          "cannot sample peak resident memory");
  require(usage.ru_maxrss >= 0 &&
              static_cast<std::uint64_t>(usage.ru_maxrss) <=
                  std::numeric_limits<std::uint64_t>::max() / 1024,
          "peak resident memory is outside the report domain");
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
}

std::pair<std::uint64_t, std::uint64_t>
selectedFifoTraversalCounts(const loom::mapping::SpatialMappingView &mapping) {
  std::uint64_t buffered = 0;
  std::uint64_t bypass = 0;
  const auto count = [&](const auto &traversal) {
    if (!traversal)
      return;
    const auto *fifo = std::get_if<loom::fabric::FabricFifoTraversalPayload>(
        &traversal->payload);
    if (!fifo)
      return;
    if (fifo->mode == loom::fabric::FabricFifoTraversalMode::Buffered)
      ++buffered;
    else
      ++bypass;
  };
  for (const auto &route : mapping.routeTrees()) {
    count(route.localTraversal);
    for (const auto &node : route.nodes)
      count(node.incomingTraversal);
    for (const auto &sink : route.sinks)
      count(sink.localTraversal);
  }
  return {buffered, bypass};
}

llvm::json::Object measurementJson(
    const loom::evaluation::models::CgraSimulationEvaluation &evaluation,
    const loom::ArtifactRootReference &evidence) {
  const std::uint64_t cycles = referenceCycles(evaluation);
  require(evaluation.attemptProfile.has_value(),
          "CGRA qualification did not collect an attempt profile");
  const auto &profile = *evaluation.attemptProfile;
  const auto &counters = profile.counters;
  require(profile.activeWallNanoseconds > 0 && counters.eventFrameCount > 0,
          "CGRA execution produced no measurable active work");
  require(profile.activeWallNanoseconds ==
              profile.inputLoadWallNanoseconds +
                  profile.engineActiveWallNanoseconds +
                  profile.observationProjectionWallNanoseconds,
          "CGRA active wall time is not its required component sum");
  llvm::json::Object result{
      {"active_wall_nanoseconds", profile.activeWallNanoseconds},
      {"input_load_process_cpu_nanoseconds",
       profile.inputLoadCpuNanoseconds
           ? llvm::json::Value(*profile.inputLoadCpuNanoseconds)
           : llvm::json::Value(nullptr)},
      {"input_load_wall_nanoseconds", profile.inputLoadWallNanoseconds},
      {"engine_active_process_cpu_nanoseconds",
       profile.engineActiveCpuNanoseconds
           ? llvm::json::Value(*profile.engineActiveCpuNanoseconds)
           : llvm::json::Value(nullptr)},
      {"engine_active_wall_nanoseconds", profile.engineActiveWallNanoseconds},
      {"observation_projection_process_cpu_nanoseconds",
       profile.observationProjectionCpuNanoseconds
           ? llvm::json::Value(*profile.observationProjectionCpuNanoseconds)
           : llvm::json::Value(nullptr)},
      {"observation_projection_wall_nanoseconds",
       profile.observationProjectionWallNanoseconds},
      {"artifact_publication_wall_nanoseconds",
       profile.artifactPublicationWallNanoseconds},
      {"artifact_publication_process_cpu_nanoseconds",
       profile.artifactPublicationCpuNanoseconds
           ? llvm::json::Value(*profile.artifactPublicationCpuNanoseconds)
           : llvm::json::Value(nullptr)},
      {"reference_cycles", cycles},
      {"event_frame_count", counters.eventFrameCount},
      {"physical_request_count", counters.physicalRequestCount},
      {"physical_grant_count", counters.physicalGrantCount},
      {"physical_retirement_count", counters.physicalRetirementCount},
      {"physical_grant_wait_cycle_sum", counters.physicalGrantWaitCycleSum},
      {"physical_grant_wait_cycle_max", counters.physicalGrantWaitCycleMax},
      {"physical_grant_delayed_count", counters.physicalGrantDelayedCount},
      {"evaluation_evidence", referenceJson(evidence)},
  };
  result["active_process_cpu_nanoseconds"] =
      profile.processCpuNanoseconds
          ? llvm::json::Value(*profile.processCpuNanoseconds)
          : llvm::json::Value(nullptr);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    llvm::errs() << "usage: " << argv[0]
                 << " ARTIFACT_STORE SOURCE_REPORT WORKLOAD_NAME OPERATOR_ID "
                    "PROTOCOL_SYMBOL\n";
    return EXIT_FAILURE;
  }

  if (llvm::Error error =
          loom::evaluation::registerProductionEvaluationRegistry())
    fail(llvm::toString(std::move(error)));
  loom::ArtifactStore artifacts(argv[1]);
  llvm::SmallString<256> blobPath(argv[1]);
  llvm::sys::path::append(blobPath, "blobs");
  loom::BlobStore blobs(blobPath);
  const SourceCase source = readSourceCase(argv[2]);
  loom::ResolvedConfig resolvedConfig = loom::defaultResolvedConfig();
  const loom::adg::BuiltinTargetScale targetScale =
      resolvedConfig.hardwareTarget.parameters;
  const loom::pnr::ResolvedPnrConfigView spatialPnrConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolvedConfig));
  const loom::mapping::ResolvedTechMappingConfigView techMappingConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolvedConfig));
  const loom::ArtifactIdentity resolvedConfigIdentity =
      take(artifacts.put(loom::ResolvedConfig::artifactSchema,
                         loom::canonicalResolvedConfigBytes(resolvedConfig)));
  const loom::ArtifactRootReference resolvedConfigReference{
      loom::ResolvedConfig::artifactSchema.identity.str(),
      loom::ResolvedConfig::artifactSchema.version, resolvedConfigIdentity};
  auto dataflow =
      take(dataflow::importCanonicalDataflow(source.dataflow, artifacts));
  const MonotonicExecutionDeadline spatialPnrDeadline(
      kSpatialPnrQualificationLimit);
  const loom::ExecutionControlView spatialPnrExecution =
      spatialPnrDeadline.control();
  auto pnrInvocation =
      take(loom::eda::test::invokeMappedBuiltinSpatialPnrFixture(
          "cgra-budget-profile", dataflow, targetScale, techMappingConfig,
          spatialPnrConfig, spatialPnrExecution, artifacts, blobs));
  llvm::json::Object techMappingResult = candidateGeneratorResultJson(
      loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor(),
      pnrInvocation.techMappingResult);
  if (!pnrInvocation.spatialPnrResult) {
    llvm::json::Object report{
        {"schema", "loom.cgra_budget_profile_outcome.2"},
        {"workload", argv[3]},
        {"operator_id", argv[4]},
        {"protocol_symbol", argv[5]},
        {"stage", "tech_mapping"},
        {"resolved_config", referenceJson(resolvedConfigReference)},
        {"fabric", referenceJson(pnrInvocation.module.reference())},
        {"tech_mapping_search", std::move(techMappingResult)},
        {"spatial_pnr", llvm::json::Value(nullptr)}};
    llvm::outs() << llvm::formatv("{0:2}\n",
                                  llvm::json::Value(std::move(report)));
    return EXIT_SUCCESS;
  }
  llvm::json::Object pnrResult =
      spatialPnrResultJson(spatialPnrConfig, *pnrInvocation.spatialPnrResult);
  const auto *completedPnr =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &pnrInvocation.spatialPnrResult->outcome);
  if (!completedPnr || completedPnr->outputBindings.front().artifacts.empty()) {
    llvm::json::Object report{
        {"schema", "loom.cgra_budget_profile_outcome.2"},
        {"workload", argv[3]},
        {"operator_id", argv[4]},
        {"protocol_symbol", argv[5]},
        {"stage", "spatial_pnr"},
        {"resolved_config", referenceJson(resolvedConfigReference)},
        {"fabric", referenceJson(pnrInvocation.module.reference())},
        {"tech_mapping_search", std::move(techMappingResult)},
        {"spatial_pnr", std::move(pnrResult)}};
    llvm::outs() << llvm::formatv("{0:2}\n",
                                  llvm::json::Value(std::move(report)));
    return EXIT_SUCCESS;
  }
  const loom::ArtifactRootReference selectedSpatialMapping =
      completedPnr->outputBindings.front().artifacts.front();
  auto importedSpatialMapping = take(
      loom::mapping::importSpatialMapping(selectedSpatialMapping, artifacts));
  const loom::ArtifactRootReference selectedTechMapping{
      loom::mapping::mappingArtifactSchema.identity.str(),
      loom::mapping::mappingArtifactSchema.version,
      importedSpatialMapping.view().techMappingIdentity()};
  auto hardware = loom::eda::test::MappedSpatialMappingFixture{
      std::move(pnrInvocation.module), selectedTechMapping,
      std::move(importedSpatialMapping)};
  const loom::ArtifactRootReference initialSpatialMapping =
      hardware.spatialMapping.reference();
  const auto [bufferedFifoTraversals, bypassFifoTraversals] =
      selectedFifoTraversalCounts(hardware.spatialMapping.view());
  llvm::errs() << "CGRA selected FIFO traversals: buffered="
               << bufferedFifoTraversals << " bypass=" << bypassFifoTraversals
               << '\n';
  const auto prepare = [&] {
    return take(loom::evaluation::models::prepareCgraSimulationEvaluation(
        source.dataflow, hardware.module.reference(),
        hardware.spatialMapping.reference(), source.workload,
        source.runtimeInput, resolvedConfig, artifacts, blobs));
  };
  auto prepared = prepare();
  const auto warmupDeadline =
      std::chrono::steady_clock::now() + kQualificationLimit;
  auto warmup =
      take(loom::evaluation::models::evaluateCgraSimulationWithAttemptProfile(
          prepared, {loom::runtime::gem5MaximumSpatialWork, warmupDeadline},
          artifacts, blobs));
  std::optional<loom::ArtifactRootReference> preRepairEvidence;
  std::optional<loom::ArtifactRootReference> parentSystemMapping;
  llvm::json::Array transportRepairAttempts;
  if (!completed(warmup)) {
    preRepairEvidence = take(loom::evaluation::publishEvaluationEvidence(
        warmup.evidence, artifacts));
    require(warmup.closedWait.has_value(),
            "incomplete CGRA warmup has no closed-wait diagnostic");
    auto system = loom::eda::test::buildMappedBuiltinSystemFixture(
        "cgra-budget-profile", targetScale, hardware.module, artifacts);
    auto systemMapping = loom::deployment::test::buildMappedSystemMapping(
        "cgra-budget-profile", dataflow, system,
        {hardware.spatialMapping.reference()}, artifacts);
    parentSystemMapping = systemMapping.reference();
    auto feedback = take(loom::dse::deriveSpatialTransportRuntimeFeedback(
        *parentSystemMapping, *warmup.closedWait, artifacts));
    require(
        feedback.disposition ==
                loom::dse::SpatialTransportRuntimeFeedbackDisposition::Exact &&
            !feedback.alternatives.empty(),
        "closed wait did not yield an exact transport repair");
    bool replayed = false;
    for (const auto &alternative : feedback.alternatives) {
      auto repaired = take(loom::eda::test::rerouteMappedSpatialMappingFixture(
          "cgra-budget-profile", dataflow, hardware, alternative,
          spatialPnrConfig, spatialPnrExecution, artifacts, blobs));
      const loom::ArtifactRootReference repairParent =
          hardware.spatialMapping.reference();
      llvm::json::Object repairPnr =
          spatialPnrResultJson(spatialPnrConfig, repaired.pnrResult);
      if (!repaired.spatialMapping) {
        transportRepairAttempts.push_back(llvm::json::Object{
            {"parent_spatial_mapping", referenceJson(repairParent)},
            {"constraint_set", referenceJson(repaired.constraintSet)},
            {"spatial_pnr", std::move(repairPnr)},
            {"child_spatial_mapping", llvm::json::Value(nullptr)},
            {"accepted_for_simulation", false}});
        continue;
      }
      auto candidateSpatial = std::move(*repaired.spatialMapping);
      const loom::ArtifactRootReference candidateReference =
          candidateSpatial.reference();
      auto candidatePrepared =
          take(loom::evaluation::models::prepareCgraSimulationEvaluation(
              source.dataflow, hardware.module.reference(),
              candidateSpatial.reference(), source.workload,
              source.runtimeInput, resolvedConfig, artifacts, blobs));
      const auto candidateDeadline =
          std::chrono::steady_clock::now() + kQualificationLimit;
      auto candidateWarmup = take(
          loom::evaluation::models::evaluateCgraSimulationWithAttemptProfile(
              candidatePrepared,
              {loom::runtime::gem5MaximumSpatialWork, candidateDeadline},
              artifacts, blobs));
      if (!completed(candidateWarmup)) {
        transportRepairAttempts.push_back(llvm::json::Object{
            {"parent_spatial_mapping", referenceJson(repairParent)},
            {"constraint_set", referenceJson(repaired.constraintSet)},
            {"spatial_pnr", std::move(repairPnr)},
            {"child_spatial_mapping", referenceJson(candidateReference)},
            {"accepted_for_simulation", false}});
        continue;
      }
      transportRepairAttempts.push_back(llvm::json::Object{
          {"parent_spatial_mapping", referenceJson(repairParent)},
          {"constraint_set", referenceJson(repaired.constraintSet)},
          {"spatial_pnr", std::move(repairPnr)},
          {"child_spatial_mapping", referenceJson(candidateReference)},
          {"accepted_for_simulation", true}});
      hardware.spatialMapping = std::move(candidateSpatial);
      prepared = std::move(candidatePrepared);
      warmup = std::move(candidateWarmup);
      replayed = true;
      break;
    }
    require(replayed, "bounded transport repair produced no retiring child");
  }
  const auto warmupEvidence = take(
      loom::evaluation::publishEvaluationEvidence(warmup.evidence, artifacts));
  (void)referenceCycles(warmup);

  llvm::json::Array measurements;
  for (std::uint64_t ordinal = 0; ordinal != kMeasurementRuns; ++ordinal) {
    const auto deadline =
        std::chrono::steady_clock::now() + kQualificationLimit;
    auto evaluated =
        take(loom::evaluation::models::evaluateCgraSimulationWithAttemptProfile(
            prepared, {loom::runtime::gem5MaximumSpatialWork, deadline},
            artifacts, blobs));
    const auto evidence = take(loom::evaluation::publishEvaluationEvidence(
        evaluated.evidence, artifacts));
    measurements.push_back(measurementJson(evaluated, evidence));
  }

  llvm::json::Object report{
      {"schema", "loom.cgra_budget_profile.5"},
      {"workload", argv[3]},
      {"operator_id", argv[4]},
      {"protocol_symbol", argv[5]},
      {"qualification_limit_nanoseconds", kQualificationLimitNanoseconds},
      {"warmup_runs", kWarmupRuns},
      {"measurement_runs", kMeasurementRuns},
      {"batch_peak_resident_bytes", peakResidentBytes()},
      {"canonical_dataflow", referenceJson(source.dataflow)},
      {"simulation_workload", referenceJson(source.workload)},
      {"simulation_runtime_input", referenceJson(source.runtimeInput)},
      {"resolved_config", referenceJson(resolvedConfigReference)},
      {"fabric", referenceJson(hardware.module.reference())},
      {"tech_mapping", referenceJson(hardware.techMapping)},
      {"tech_mapping_search", std::move(techMappingResult)},
      {"initial_spatial_mapping", referenceJson(initialSpatialMapping)},
      {"spatial_mapping", referenceJson(hardware.spatialMapping.reference())},
      {"spatial_pnr", std::move(pnrResult)},
      {"transport_repair",
       parentSystemMapping && preRepairEvidence
           ? llvm::json::Value(llvm::json::Object{
                 {"parent_system_mapping", referenceJson(*parentSystemMapping)},
                 {"pre_repair_evidence", referenceJson(*preRepairEvidence)},
                 {"attempts", std::move(transportRepairAttempts)}})
           : llvm::json::Value(nullptr)},
      {"warmup_evidence", referenceJson(warmupEvidence)},
      {"measurements", std::move(measurements)},
  };
  llvm::outs() << llvm::formatv("{0:2}\n",
                                llvm::json::Value(std::move(report)));
  return EXIT_SUCCESS;
}
