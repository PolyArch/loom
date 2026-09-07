#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/ExecutionControl.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Common/TimeoutBudgets.h"
#include "Config/ResolvedConfig.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialTransportCegar.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/TechMappingHardwareFeedback.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraClosedWait.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricEnums.h"
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

constexpr llvm::StringLiteral kHardwareSearchSchema =
    "loom.cgra_qualification_hardware_search.1";

constexpr llvm::StringLiteral kProfileSchema = "loom.cgra_budget_profile.6";
constexpr llvm::StringLiteral kProfileOutcomeSchema =
    "loom.cgra_budget_profile_outcome.3";

constexpr std::uint64_t kWarmupRuns = 1;
constexpr std::uint64_t kMeasurementRuns = 3;
constexpr std::uint64_t kQualificationLimitNanoseconds = 45'000'000'000ULL;
constexpr auto kQualificationLimit = std::chrono::seconds(45);
constexpr auto kSpatialPnrQualificationLimit =
    loom::timeout::duration(loom::timeout::Tier::Fast);
constexpr auto kTransportRepairQualificationLimit =
    loom::timeout::duration(loom::timeout::Tier::Long);
constexpr std::uint64_t kTransportRepairMaximumIterations = 8;
/// A healthy Matmul warmup retires in well under one second, so screening the
/// published Spatial frontier costs a small fraction of one qualification
/// deadline while a non-retiring candidate still reaches its closed-wait
/// diagnostic inside this window.
constexpr auto kCandidateScreeningLimit = std::chrono::seconds(10);

/// Mutually exclusive wall and process-CPU spans of the qualification. Every
/// phase closes before the next opens, so the recorded spans sum to the
/// measured total rather than to a parallel accumulation.
class PhaseLedger final {
public:
  void record(llvm::StringRef phase) {
    const auto now = std::chrono::steady_clock::now();
    const std::uint64_t cpu = processCpuNanoseconds();
    entries_.push_back(llvm::json::Object{
        {"phase", phase.str()},
        {"wall_nanoseconds",
         static_cast<std::uint64_t>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark_)
                 .count())},
        {"process_cpu_nanoseconds", cpu - cpuMark_}});
    mark_ = now;
    cpuMark_ = cpu;
  }

  llvm::json::Array release() { return std::move(entries_); }

private:
  static std::uint64_t processCpuNanoseconds() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
      return 0;
    const auto convert = [](const timeval &value) -> std::uint64_t {
      return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
             static_cast<std::uint64_t>(value.tv_usec) * 1'000ULL;
    };
    return convert(usage.ru_utime) + convert(usage.ru_stime);
  }

  std::chrono::steady_clock::time_point mark_ =
      std::chrono::steady_clock::now();
  std::uint64_t cpuMark_ = processCpuNanoseconds();
  llvm::json::Array entries_;
};

class MonotonicExecutionDeadline final {
public:
  explicit MonotonicExecutionDeadline(
      std::chrono::steady_clock::duration duration)
      : notAfter_(std::chrono::steady_clock::now() + duration) {}

  loom::ExecutionControlView control() const {
    return {this, stopRequested, remainingTime};
  }

  /// Nanoseconds the observing owner ran past the deadline; zero while the
  /// deadline has not been reached. Measured when the owner returns.
  std::uint64_t overrunNanoseconds() const {
    const auto now = std::chrono::steady_clock::now();
    if (now <= notAfter_)
      return 0;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - notAfter_)
            .count());
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

loom::ResolvedConfig qualificationConfig() {
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  const auto &target = loom::adg::builtinLargeTarget;
  auto scale = target.scale;
  scale.temporalResidentContexts = 16;
  config.hardwareTarget = {target.templateIdentity.str(),
                           {target.schemaMajor, target.schemaMinor},
                           scale};
  return config;
}

loom::ArtifactRootReference publishConfig(const loom::ResolvedConfig &config,
                                          loom::ArtifactStore &artifacts) {
  return {loom::ResolvedConfig::artifactSchema.identity.str(),
          loom::ResolvedConfig::artifactSchema.version,
          take(artifacts.put(loom::ResolvedConfig::artifactSchema,
                             loom::canonicalResolvedConfigBytes(config)))};
}

llvm::json::Value readJson(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    fail("cannot read qualification input: " + buffer.getError().message());
  return take(llvm::json::parse((*buffer)->getBuffer()));
}

struct QualificationSource final {
  std::string workload;
  std::string operatorId;
  std::string protocolSymbol;
  SourceCase source;
};

llvm::json::Object sourceIdentityJson(const QualificationSource &source) {
  return llvm::json::Object{
      {"workload", source.workload},
      {"operator_id", source.operatorId},
      {"protocol_symbol", source.protocolSymbol},
      {"canonical_dataflow", referenceJson(source.source.dataflow)},
      {"simulation_workload", referenceJson(source.source.workload)},
      {"simulation_runtime_input", referenceJson(source.source.runtimeInput)}};
}

llvm::json::Object selectQualificationHardware(
    llvm::StringRef requestPath, const loom::ResolvedConfig &config,
    const loom::ArtifactRootReference &configReference,
    loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  auto requestDocument = readJson(requestPath);
  const auto *requests = requestDocument.getAsArray();
  require(requests && !requests->empty(),
          "qualification hardware requires the source suite");
  std::vector<QualificationSource> sources;
  for (const auto &value : *requests) {
    const auto *request = value.getAsObject();
    require(request && request->size() == 4 && request->getString("workload") &&
                request->getString("operator_id") &&
                request->getString("protocol_symbol") &&
                request->getString("source_report"),
            "qualification source request is malformed");
    const auto name = *request->getString("workload");
    require(llvm::none_of(
                sources,
                [&](const auto &source) { return source.workload == name; }),
            "qualification source repeats a workload");
    sources.push_back({name.str(), request->getString("operator_id")->str(),
                       request->getString("protocol_symbol")->str(),
                       readSourceCase(*request->getString("source_report"))});
  }

  const MonotonicExecutionDeadline deadline(kSpatialPnrQualificationLimit);
  const auto control = deadline.control();
  PhaseLedger ledger;
  loom::adg::DesignBuilder builder(artifacts);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      builder, config.hardwareTarget.parameters));
  if (auto error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  require(design.roots().size() == 1,
          "qualification hardware did not produce one module");
  auto module = design.roots().front();
  const auto initialFabric = module.reference();
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(config));
  const auto techBinding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          techConfig));
  if (auto error =
          loom::dse::registerSpatialMicroarchitectureCandidateGenerator())
    fail(llvm::toString(std::move(error)));
  llvm::json::Array rounds;
  bool ready = false;
  for (;;) {
    llvm::json::Array evaluations;
    std::optional<loom::mapping::TechMappingComputeContextHallDeficit> pressure;
    ready = true;
    for (const auto &source : sources) {
      auto inputs =
          take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
              {source.source.dataflow}, module.reference()));
      auto result = take(loom::dse::invokeCandidateGenerator(
          inputs, techBinding, artifacts, blobs, control));
      const bool mapped = !candidateArtifacts(result).empty();
      const auto *incomplete =
          std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
              &result.outcome);
      ready &= mapped && (!incomplete ||
                          incomplete->reason ==
                              loom::dse::CandidateGeneratorIncompleteReason::
                                  SemanticLimitReached);
      auto evaluation = sourceIdentityJson(source);
      evaluation["tech_mapping_search"] = candidateGeneratorResultJson(
          loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor(),
          result);
      evaluation["owner_feedback"] =
          result.ownerFeedback
              ? llvm::json::Value(llvm::toHex(*result.ownerFeedback, true))
              : llvm::json::Value(nullptr);
      if (!mapped && result.ownerFeedback)
        loom::mapping::retainTechMappingComputeContextHallFeedback(
            pressure,
            take(loom::mapping::adoptTechMappingComputeContextHallFeedback(
                *result.ownerFeedback, module.view())));
      evaluations.push_back(std::move(evaluation));
    }
    llvm::json::Object round{{"fabric", referenceJson(module.reference())},
                             {"evaluations", std::move(evaluations)},
                             {"hardware_growth", nullptr}};
    if (ready || !pressure || control.stopRequested()) {
      rounds.push_back(std::move(round));
      break;
    }
    auto growth =
        take(loom::dse::projectTechMappingComputeContextJointGrowthPlan(
            *pressure, module.view()));
    require(growth.addedContextCount > 0 && !growth.decisions.empty(),
            "qualification hardware feedback made no progress");
    const loom::dse::SpatialMicroarchitectureDecisionDomain domain =
        loom::dse::ResizeInstructionStoresDomain{growth.decisions};
    auto growthConfig = take(
        loom::dse::resolveSpatialMicroarchitectureRewriteConfig({domain}, 1));
    auto growthBinding = take(
        loom::dse::resolveSpatialMicroarchitectureCandidateGeneratorBinding(
            growthConfig));
    auto growthInputs =
        take(loom::dse::bindSpatialMicroarchitectureCandidateGeneratorInputs(
            {module.reference()}));
    auto result = take(loom::dse::invokeCandidateGenerator(
        growthInputs, growthBinding, artifacts, blobs, control));
    round["hardware_growth"] = llvm::json::Object{
        {"owner_feedback",
         llvm::toHex(loom::mapping::encodeTechMappingComputeContextHallFeedback(
                         *pressure),
                     true)},
        {"canonical_config",
         llvm::toHex(growthBinding.canonicalConfigBytes(), true)},
        {"result",
         candidateGeneratorResultJson(
             loom::dse::spatialMicroarchitectureCandidateGeneratorDescriptor(),
             result)}};
    rounds.push_back(std::move(round));
    if (!std::holds_alternative<loom::dse::CompletedCandidateGeneratorResult>(
            result.outcome))
      break;
    require(candidateArtifacts(result).size() == 1,
            "qualification hardware did not publish one atomic child");
    module = take(loom::fabric::importEntireFabricRoot(
        candidateArtifacts(result).front(), artifacts));
  }
  ledger.record("shared_hardware_search");
  return llvm::json::Object{
      {"schema", kHardwareSearchSchema},
      {"resolved_config", referenceJson(configReference)},
      {"initial_fabric", referenceJson(initialFabric)},
      {"fabric", referenceJson(module.reference())},
      {"ready", ready && !control.stopRequested()},
      {"deadline_ns", static_cast<std::uint64_t>(
                          std::chrono::duration_cast<std::chrono::nanoseconds>(
                              kSpatialPnrQualificationLimit)
                              .count())},
      {"deadline_overrun_ns", deadline.overrunNanoseconds()},
      {"rounds", std::move(rounds)},
      {"phase_ledger", ledger.release()}};
}

loom::fabric::FinalizedFabricRoot
readQualificationHardware(llvm::StringRef path,
                          const QualificationSource &source,
                          const loom::ArtifactRootReference &configReference,
                          const loom::ArtifactStore &artifacts) {
  auto document = readJson(path);
  const auto *report = document.getAsObject();
  require(report && report->getString("schema") == kHardwareSearchSchema &&
              report->getBoolean("ready") == true &&
              report->getObject("resolved_config") &&
              report->getObject("fabric") && report->getArray("rounds"),
          "qualification hardware search is incomplete");
  require(take(loom::parseArtifactRootReferenceJson(
              *report->getObject("resolved_config"))) == configReference,
          "qualification hardware uses a foreign resolved config");
  const auto *round = report->getArray("rounds")->empty()
                          ? nullptr
                          : report->getArray("rounds")->back().getAsObject();
  const auto *evaluations = round ? round->getArray("evaluations") : nullptr;
  require(evaluations, "qualification hardware omits the source suite");
  const auto identity = sourceIdentityJson(source);
  require(llvm::any_of(*evaluations,
                       [&](const auto &value) {
                         const auto *evaluation = value.getAsObject();
                         if (!evaluation)
                           return false;
                         for (const auto &field : identity) {
                           const auto *observed = evaluation->get(field.first);
                           if (!observed || *observed != field.second)
                             return false;
                         }
                         return true;
                       }),
          "qualification profile is absent from the hardware source suite");
  return take(loom::fabric::importEntireFabricRoot(
      take(loom::parseArtifactRootReferenceJson(*report->getObject("fabric"))),
      artifacts));
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
  llvm::errs() << "CGRA wait certificate: closed="
               << loom::sim::verifyClosedWaitCertificateClosure(diagnostic)
               << " edges=" << diagnostic.waitCertificate.size();
  if (diagnostic.waitProofFailure)
    llvm::errs() << " proof_failure="
                 << static_cast<unsigned>(*diagnostic.waitProofFailure);
  llvm::errs() << '\n';
  const auto ownerText = [](llvm::raw_ostream &out,
                            const loom::sim::CgraClosedWaitSetDiagnostic::
                                WaitOwnerKey &owner) {
    using Diagnostic = loom::sim::CgraClosedWaitSetDiagnostic;
    if (const auto *firing = std::get_if<0>(&owner.owner)) {
      out << "actor:" << firing->semanticActorOrdinal << "/"
          << firing->occurrenceOrdinal;
      return;
    }
    const auto &queue = std::get<1>(owner.owner);
    out << (queue.domain == Diagnostic::WaitStorageDomain::TraversalStorage
                ? "storage:"
                : "operand_queue:")
        << queue.ordinal
        << (queue.queueClass.tagLocal ? "/tag:" : "/global");
    if (queue.queueClass.tagLocal) {
      llvm::SmallString<24> text;
      queue.queueClass.tagValue.toStringUnsigned(text, 10);
      out << text;
    }
  };
  for (const auto &edge : diagnostic.waitCertificate) {
    llvm::errs() << "CGRA wait certificate edge: ";
    ownerText(llvm::errs(), edge.from);
    llvm::errs() << " -> ";
    ownerText(llvm::errs(), edge.to);
    llvm::errs() << " kind=" << static_cast<unsigned>(edge.kind)
                 << " binding=" << edge.bindingOrdinal
                 << " occurrence=" << edge.occurrenceOrdinal
                 << " awaited_class_position=" << edge.awaitedClassPosition;
    if (edge.headTagValue) {
      llvm::SmallString<24> text;
      edge.headTagValue->toStringUnsigned(text, 10);
      llvm::errs() << " head_tag=" << text;
    }
    if (edge.awaitedTagValue) {
      llvm::SmallString<24> text;
      edge.awaitedTagValue->toStringUnsigned(text, 10);
      llvm::errs() << " awaited_tag=" << text;
    }
    llvm::errs() << " head_binding=" << edge.headBindingOrdinal << '\n';
  }
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
             evaluation.evidence.outcome()) &&
         !evaluation.closedWait;
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
  const bool selectHardware = argc == 4 && llvm::StringRef(argv[1]) == "--hardware";
  if (!selectHardware && argc != 7) {
    llvm::errs() << "usage: " << argv[0]
                 << " --hardware ARTIFACT_STORE SOURCE_REQUESTS\n       "
                 << argv[0]
                 << " ARTIFACT_STORE SOURCE_REPORT WORKLOAD_NAME OPERATOR_ID "
                    "PROTOCOL_SYMBOL HARDWARE_SEARCH_REPORT\n";
    return EXIT_FAILURE;
  }

  if (llvm::Error error =
          loom::evaluation::registerProductionEvaluationRegistry())
    fail(llvm::toString(std::move(error)));
  PhaseLedger ledger;
  const char *storePath = argv[selectHardware ? 2 : 1];
  loom::ArtifactStore artifacts(storePath);
  llvm::SmallString<256> blobPath(storePath);
  llvm::sys::path::append(blobPath, "blobs");
  loom::BlobStore blobs(blobPath);
  const loom::ResolvedConfig resolvedConfig = qualificationConfig();
  const auto resolvedConfigReference = publishConfig(resolvedConfig, artifacts);
  if (selectHardware) {
    auto report = selectQualificationHardware(
        argv[3], resolvedConfig, resolvedConfigReference, artifacts, blobs);
    llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(report)));
    return EXIT_SUCCESS;
  }
  const SourceCase source = readSourceCase(argv[2]);
  auto module = readQualificationHardware(
      argv[6], {argv[3], argv[4], argv[5], source}, resolvedConfigReference,
      artifacts);
  const auto targetScale = resolvedConfig.hardwareTarget.parameters;
  const auto spatialPnrConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolvedConfig));
  const auto techMappingConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolvedConfig));
  auto dataflow =
      take(dataflow::importCanonicalDataflow(source.dataflow, artifacts));
  const MonotonicExecutionDeadline spatialPnrDeadline(
      kSpatialPnrQualificationLimit);
  const loom::ExecutionControlView spatialPnrExecution =
      spatialPnrDeadline.control();
  ledger.record("setup");
  auto pnrInvocation =
      take(loom::eda::test::invokeMappedSpatialPnrFixture(
          "cgra-budget-profile", dataflow, std::move(module), techMappingConfig,
          spatialPnrConfig, spatialPnrExecution, artifacts, blobs));
  const std::uint64_t spatialPnrDeadlineOverrun =
      spatialPnrDeadline.overrunNanoseconds();
  llvm::json::Object techMappingResult = candidateGeneratorResultJson(
      loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor(),
      pnrInvocation.techMappingResult);
  if (!pnrInvocation.spatialPnrResult) {
    llvm::json::Object report{
        {"schema", kProfileOutcomeSchema},
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
  pnrResult["deadline_ns"] = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          kSpatialPnrQualificationLimit)
          .count());
  pnrResult["deadline_overrun_ns"] = spatialPnrDeadlineOverrun;
  const std::vector<loom::ArtifactRootReference> &publishedSpatialMappings =
      candidateArtifacts(*pnrInvocation.spatialPnrResult);
  if (publishedSpatialMappings.empty()) {
    llvm::json::Object report{
        {"schema", kProfileOutcomeSchema},
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
  ledger.record("spatial_pnr");
  // One exact Fabric owner is shared by every Mapping in this invocation.
  // Nested strict imports reuse this bounded session; dynamic executions and
  // Mapping-specific preparations keep their existing separate lifetimes.
  loom::fabric::FabricArtifactImportSession fabricImports;
  // Every published Spatial candidate cost a complete restart, and the
  // published order is canonical artifact identity, not quality. Screen the
  // whole frontier against the one dynamic oracle that the static Mapping
  // model does not decide, and retain the first candidate that retires.
  llvm::json::Array candidateScreening;
  std::optional<loom::eda::test::MappedSpatialMappingFixture> selectedHardware;
  std::optional<loom::evaluation::models::PreparedCgraSimulationEvaluation>
      selectedPrepared;
  std::optional<loom::evaluation::models::CgraSimulationEvaluation>
      selectedWarmup;
  std::optional<loom::eda::test::MappedSpatialMappingFixture> repairHardware;
  std::optional<loom::evaluation::models::PreparedCgraSimulationEvaluation>
      repairPrepared;
  std::optional<loom::evaluation::models::CgraSimulationEvaluation>
      repairWarmup;
  std::optional<std::pair<std::uint64_t, std::uint64_t>> repairScore;
  for (const loom::ArtifactRootReference &candidate :
       publishedSpatialMappings) {
    PhaseLedger screeningPhases;
    auto imported =
        take(loom::mapping::importSpatialMapping(candidate, artifacts));
    const loom::ArtifactRootReference candidateTechMapping{
        loom::mapping::mappingArtifactSchema.identity.str(),
        loom::mapping::mappingArtifactSchema.version,
        imported.view().techMappingIdentity()};
    const auto [buffered, bypass] =
        selectedFifoTraversalCounts(imported.view());
    auto candidateHardware = loom::eda::test::MappedSpatialMappingFixture{
        pnrInvocation.module, candidateTechMapping, std::move(imported)};
    screeningPhases.record("mapping_import");
    auto candidatePrepared =
        take(loom::evaluation::models::prepareCgraSimulationEvaluation(
            source.dataflow, candidateHardware.module.reference(),
            candidateHardware.spatialMapping.reference(), source.workload,
            source.runtimeInput, resolvedConfig, artifacts, blobs));
    screeningPhases.record("preparation");
    const bool last = candidate == publishedSpatialMappings.back();
    const auto screeningDeadline =
        std::chrono::steady_clock::now() + (selectedWarmup || !last
                                                ? kCandidateScreeningLimit
                                                : kQualificationLimit);
    auto screened =
        take(loom::evaluation::models::evaluateCgraSimulationWithAttemptProfile(
            candidatePrepared,
            {loom::runtime::gem5MaximumSpatialWork, screeningDeadline},
            artifacts, blobs));
    screeningPhases.record("evaluation");
    loom::emitInvocationDiagnostic(
        loom::DiagnosticVerbosity::Summary,
        loom::InvocationDiagnosticStage::SystemPnr,
        loom::InvocationDiagnosticEvent::Statistics, [&] {
          llvm::json::Object fields{
              {"operation", "cgra_candidate_screening"},
              {"spatial_mapping", referenceJson(candidate)},
              {"phase_ledger", screeningPhases.release()}};
          if (screened.attemptProfile) {
            const auto &profile = *screened.attemptProfile;
            fields["active_wall_nanoseconds"] = profile.activeWallNanoseconds;
            fields["engine_active_wall_nanoseconds"] =
                profile.engineActiveWallNanoseconds;
            fields["artifact_publication_wall_nanoseconds"] =
                profile.artifactPublicationWallNanoseconds;
          }
          return llvm::json::Value(std::move(fields));
        });
    const bool retired = completed(screened);
    if (!retired)
      llvm::errs() << "CGRA screening outcome: "
                   << loom::evaluation::toString(screened.evidence.outcomeKind())
                   << " event_frames="
                   << (screened.attemptProfile
                           ? screened.attemptProfile->counters.eventFrameCount
                           : 0)
                   << " actor_retirements="
                   << (screened.attemptProfile
                           ? screened.attemptProfile->counters
                                 .actorRetirementCount
                           : 0)
                   << " publications="
                   << (screened.attemptProfile
                           ? screened.attemptProfile->counters
                                 .tokenPublicationCount
                           : 0)
                   << '\n';
    if (screened.closedWait)
      emitClosedWaitDiagnostic(*screened.closedWait);
    candidateScreening.push_back(llvm::json::Object{
        {"spatial_mapping", referenceJson(candidate)},
        {"buffered_fifo_traversals", buffered},
        {"bypass_fifo_traversals", bypass},
        {"retired", retired},
        {"closed_wait_actor_cycle_edges",
         screened.closedWait ? llvm::json::Value(static_cast<std::uint64_t>(
                                   screened.closedWait->actorWaitCycle.size()))
                             : llvm::json::Value(nullptr)},
        {"closed_wait_pending_transfers",
         screened.closedWait
             ? llvm::json::Value(screened.closedWait->pendingTransfers)
             : llvm::json::Value(nullptr)},
        {"closed_wait_certificate_edges",
         screened.closedWait
             ? llvm::json::Value(static_cast<std::uint64_t>(
                   screened.closedWait->waitCertificate.size()))
             : llvm::json::Value(nullptr)},
        {"closed_wait_certificate_closed",
         screened.closedWait
             ? llvm::json::Value(
                   loom::sim::verifyClosedWaitCertificateClosure(
                       *screened.closedWait))
             : llvm::json::Value(nullptr)},
        {"closed_wait_proof_failure",
         screened.closedWait && screened.closedWait->waitProofFailure
             ? llvm::json::Value(static_cast<std::uint64_t>(
                   *screened.closedWait->waitProofFailure))
             : llvm::json::Value(nullptr)},
        {"operand_queue_shared_ingress_pressure",
         screened.closedWait
             ? llvm::json::Value(
                   screened.closedWait->operandQueueSharedIngressPressure)
             : llvm::json::Value(nullptr)}});
    if (retired && !selectedWarmup) {
      selectedHardware = std::move(candidateHardware);
      selectedPrepared = std::move(candidatePrepared);
      selectedWarmup = std::move(screened);
    } else if (!retired && screened.closedWait &&
               !screened.closedWait->waitProofFailure &&
               loom::sim::verifyClosedWaitCertificateClosure(
                   *screened.closedWait)) {
      const std::pair<std::uint64_t, std::uint64_t> score{
          screened.closedWait->waitCertificate.size(),
          screened.closedWait->pendingTransfers};
      if (!repairScore || score < *repairScore) {
        repairScore = score;
        repairHardware = std::move(candidateHardware);
        repairPrepared = std::move(candidatePrepared);
        repairWarmup = std::move(screened);
      }
    }
  }
  ledger.record("candidate_screening");
  loom::fabric::emitFabricArtifactImportSessionStatistics(
      loom::fabric::FabricArtifactImportVerificationDomain::SourceInvocation,
      loom::InvocationDiagnosticStage::SystemPnr, fabricImports.statistics());
  require(selectedHardware || repairHardware,
          "screened Spatial frontier has no retiring or proven closed-wait "
          "candidate");
  // A retiring Mapping remains the first choice. Otherwise repair the
  // deterministic smallest closed certificate; ties retain canonical
  // publication order.
  auto hardware = selectedHardware ? std::move(*selectedHardware)
                                   : std::move(*repairHardware);
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
  auto prepared = selectedPrepared ? std::move(*selectedPrepared)
                                   : std::move(*repairPrepared);
  auto warmup = selectedWarmup ? std::move(*selectedWarmup)
                               : std::move(*repairWarmup);
  std::optional<loom::ArtifactRootReference> preRepairEvidence;
  std::optional<loom::ArtifactRootReference> parentSystemMapping;
  std::optional<loom::dse::SpatialTransportCegarTermination> repairTermination;
  llvm::json::Array transportRepairAttempts;
  if (!completed(warmup)) {
    preRepairEvidence = take(loom::evaluation::publishEvaluationEvidence(
        warmup.evidence, artifacts));
    llvm::errs() << "CGRA warmup outcome: "
                 << loom::evaluation::toString(warmup.evidence.outcomeKind())
                 << " event_frames="
                 << (warmup.attemptProfile
                         ? warmup.attemptProfile->counters.eventFrameCount
                         : 0)
                 << " actor_retirements="
                 << (warmup.attemptProfile
                         ? warmup.attemptProfile->counters.actorRetirementCount
                         : 0)
                 << " publications="
                 << (warmup.attemptProfile
                         ? warmup.attemptProfile->counters.tokenPublicationCount
                         : 0)
                 << '\n';
    require(warmup.closedWait.has_value(),
            "incomplete CGRA warmup has no closed-wait diagnostic");
    auto system = loom::eda::test::buildMappedBuiltinSystemFixture(
        "cgra-budget-profile", targetScale, hardware.module, artifacts);
    auto systemMapping = loom::deployment::test::buildMappedSystemMapping(
        "cgra-budget-profile", dataflow, system,
        {hardware.spatialMapping.reference()}, artifacts);
    parentSystemMapping = systemMapping.reference();
    const auto &dataflowView = dataflow.view();
    auto techMapping =
        take(loom::mapping::importTechMapping(hardware.techMapping, artifacts));
    auto parentConstraints =
        take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
            dataflowView, techMapping.view(), hardware.module.view(),
            artifacts));
    auto verifiedWait = take(
        loom::evaluation::models::importVerifiedCgraClosedWaitEvidence(
            *preRepairEvidence, artifacts, blobs));
    auto physicalTiming =
        take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
            hardware.module.view()));
    const auto cegarDeadline =
        std::chrono::steady_clock::now() +
        kTransportRepairQualificationLimit;
    auto cegar = take(loom::dse::executeSpatialTransportCegar(
        hardware.spatialMapping.reference(), parentConstraints.reference(),
        verifiedWait, resolvedConfig, physicalTiming,
        {kTransportRepairMaximumIterations,
         kTransportRepairMaximumIterations,
         spatialPnrConfig.policy().search.exactRepair.maxSolverCalls,
         loom::runtime::gem5MaximumSpatialWork,
         cegarDeadline},
        artifacts, blobs));
    repairTermination = cegar.termination;
    llvm::errs() << "CGRA CEGAR termination: "
                 << loom::dse::spatialTransportCegarTerminationSpelling(
                        cegar.termination)
                 << " iterations=" << cegar.iterations.size() << '\n';
    for (const auto &iteration : cegar.iterations)
      llvm::errs() << "CGRA CEGAR iteration: repair_kind="
                   << static_cast<std::uint64_t>(iteration.repair.kind)
                   << " logical_solver_calls="
                   << iteration.repair.logicalSolverCalls
                   << " endpoint_expansions="
                   << iteration.repair.endpointExpansions
                   << " negotiation_iterations="
                   << iteration.repair.negotiationIterations
                   << " actions=" << iteration.repair.actionCount
                   << " retired=" << iteration.retired
                   << " promotion_wall_ns="
                   << iteration.work.promotion.activeWallTimeNanoseconds
                   << " freeze_wall_ns="
                   << iteration.work.problemFreeze.activeWallTimeNanoseconds
                   << " warm_seed_wall_ns="
                   << iteration.work.warmSeed.activeWallTimeNanoseconds
                   << " exact_repair_wall_ns="
                   << iteration.work.exactRepair.activeWallTimeNanoseconds
                   << " finalization_wall_ns="
                   << iteration.work.childFinalization.activeWallTimeNanoseconds
                   << " runtime_wall_ns="
                   << iteration.work.runtimeEvaluation.activeWallTimeNanoseconds
                   << " evidence_verification_wall_ns="
                   << iteration.work.evidenceVerification
                          .activeWallTimeNanoseconds
                   << '\n';
    for (const auto &iteration : cegar.iterations)
      transportRepairAttempts.push_back(llvm::json::Object{
          {"parent_spatial_mapping", referenceJson(iteration.parentMapping)},
          {"runtime_evidence", referenceJson(iteration.runtimeEvidence)},
          {"constraint_set", referenceJson(iteration.accumulatedConstraints)},
          {"child_spatial_mapping",
           iteration.childMapping
               ? llvm::json::Value(referenceJson(*iteration.childMapping))
               : llvm::json::Value(nullptr)},
          {"child_evidence",
           iteration.childEvidence
               ? llvm::json::Value(referenceJson(*iteration.childEvidence))
               : llvm::json::Value(nullptr)},
          {"repair_kind", static_cast<std::uint64_t>(iteration.repair.kind)},
          {"solver_calls", iteration.repair.solverCalls},
          {"logical_solver_calls", iteration.repair.logicalSolverCalls},
          {"action_count", iteration.repair.actionCount},
          {"retired", iteration.retired}});
    const bool replayed =
        cegar.termination ==
            loom::dse::SpatialTransportCegarTermination::Retired &&
        cegar.finalMapping.has_value();
    if (replayed) {
      hardware.spatialMapping = take(loom::mapping::importSpatialMapping(
          *cegar.finalMapping, artifacts));
      prepared = prepare();
      warmup = take(
          loom::evaluation::models::evaluateCgraSimulationWithAttemptProfile(
              prepared,
              {loom::runtime::gem5MaximumSpatialWork,
               std::chrono::steady_clock::now() + kQualificationLimit},
              artifacts, blobs));
      require(completed(warmup),
              "CEGAR-retired child did not replay as a retired Mapping");
    }
    ledger.record("transport_repair");
    if (!replayed) {
      // A qualification that cannot retire still owns complete evidence: the
      // screened frontier, every repair attempt, and the phase ledger. Exiting
      // without it would discard work that was already paid for.
      llvm::json::Object report{
          {"schema", kProfileOutcomeSchema},
          {"workload", argv[3]},
          {"operator_id", argv[4]},
          {"protocol_symbol", argv[5]},
          {"stage", "transport_repair"},
          {"resolved_config", referenceJson(resolvedConfigReference)},
          {"fabric", referenceJson(pnrInvocation.module.reference())},
          {"tech_mapping_search", std::move(techMappingResult)},
          {"spatial_pnr", std::move(pnrResult)},
          {"initial_spatial_mapping", referenceJson(initialSpatialMapping)},
          {"spatial_candidate_screening", std::move(candidateScreening)},
          {"transport_repair",
           llvm::json::Object{
               {"parent_system_mapping", referenceJson(*parentSystemMapping)},
               {"pre_repair_evidence", referenceJson(*preRepairEvidence)},
               {"termination",
                loom::dse::spatialTransportCegarTerminationSpelling(
                    *repairTermination)},
               {"attempts", std::move(transportRepairAttempts)}}},
          {"phase_ledger", ledger.release()}};
      llvm::outs() << llvm::formatv("{0:2}\n",
                                    llvm::json::Value(std::move(report)));
      return EXIT_SUCCESS;
    }
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

  ledger.record("measurements");
  llvm::json::Object report{
      {"schema", kProfileSchema},
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
      {"spatial_candidate_screening", std::move(candidateScreening)},
      {"spatial_mapping", referenceJson(hardware.spatialMapping.reference())},
      {"spatial_pnr", std::move(pnrResult)},
      {"transport_repair",
       parentSystemMapping && preRepairEvidence
           ? llvm::json::Value(llvm::json::Object{
                 {"parent_system_mapping", referenceJson(*parentSystemMapping)},
                 {"pre_repair_evidence", referenceJson(*preRepairEvidence)},
                 {"termination",
                  loom::dse::spatialTransportCegarTerminationSpelling(
                      *repairTermination)},
                 {"attempts", std::move(transportRepairAttempts)}})
           : llvm::json::Value(nullptr)},
      {"warmup_evidence", referenceJson(warmupEvidence)},
      {"measurements", std::move(measurements)},
      {"phase_ledger", ledger.release()},
  };
  llvm::outs() << llvm::formatv("{0:2}\n",
                                llvm::json::Value(std::move(report)));
  return EXIT_SUCCESS;
}
