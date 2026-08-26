#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/ExecutionControl.h"
#include "Config/ResolvedConfig.h"
#include "DSE/SpatialRuntimeFeedback.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/PnrConfig.h"
#include "Runtime/Gem5DispatchABI.h"

#include "MappedRtlSimulationTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/resource.h>

#include <algorithm>
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
constexpr auto kSpatialPnrLimit = std::chrono::seconds(240);
constexpr std::uint32_t kSpatialPnrProposalsPerLevel = 128;
constexpr std::uint32_t kQualificationMeshDimension = 12;
constexpr std::uint32_t kQualificationSpatialMeshLanesPerDirection = 4;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "CGRA budget profile: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

struct DeadlineControl final {
  std::chrono::steady_clock::time_point deadline;

  static bool stopRequested(const void *context) {
    return std::chrono::steady_clock::now() >=
           static_cast<const DeadlineControl *>(context)->deadline;
  }

  static std::optional<std::chrono::steady_clock::duration>
  remainingTime(const void *context) {
    const auto remaining =
        static_cast<const DeadlineControl *>(context)->deadline -
        std::chrono::steady_clock::now();
    return std::max(remaining, std::chrono::steady_clock::duration::zero());
  }

  loom::ExecutionControlView view() const {
    return {this, stopRequested, remainingTime};
  }
};

loom::adg::BuiltinTargetScale qualificationTargetScale() {
  loom::adg::BuiltinTargetScale scale = loom::adg::builtinLargeTarget.scale;
  scale.meshDimension = kQualificationMeshDimension;
  scale.spatialMeshLanesPerDirection =
      kQualificationSpatialMeshLanesPerDirection;
  scale.spatialFuOccurrences = {scale.spatialPeCount, scale.spatialPeCount,
                                scale.spatialPeCount, scale.spatialPeCount,
                                scale.spatialPeCount, scale.spatialPeCount,
                                scale.spatialPeCount, scale.spatialPeCount};
  scale.temporalFuOccurrences = {scale.temporalPeCount, scale.temporalPeCount,
                                 scale.temporalPeCount, scale.temporalPeCount,
                                 scale.temporalPeCount, scale.temporalPeCount,
                                 scale.temporalPeCount, scale.temporalPeCount};
  return scale;
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

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  ::mapping::MappingDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
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
  const auto &profile = evaluation.attemptProfile;
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
  if (argc != 5) {
    llvm::errs() << "usage: " << argv[0]
                 << " ARTIFACT_STORE SOURCE_REPORT WORKLOAD_NAME OPERATOR_ID\n";
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
  const loom::adg::BuiltinTargetScale targetScale = qualificationTargetScale();
  loom::ResolvedConfig resolvedConfig =
      take(loom::resolveConfigProfile("quick_explore"));
  resolvedConfig.hardwareTarget.parameters = targetScale;
  auto &spatialAnnealing = resolvedConfig.dse.spatialPnr.search.annealing;
  spatialAnnealing.coolingRatio = {1, 2};
  spatialAnnealing.proposalsPerLevelBase = kSpatialPnrProposalsPerLevel;
  spatialAnnealing.proposalsPerMovableDecision = 0;
  resolvedConfig.dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  const loom::pnr::ResolvedPnrConfigView spatialPnrConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolvedConfig));
  const loom::ArtifactIdentity resolvedConfigIdentity =
      take(artifacts.put(loom::ResolvedConfig::artifactSchema,
                         loom::canonicalResolvedConfigBytes(resolvedConfig)));
  const loom::ArtifactRootReference resolvedConfigReference{
      loom::ResolvedConfig::artifactSchema.identity.str(),
      loom::ResolvedConfig::artifactSchema.version, resolvedConfigIdentity};
  auto dataflow =
      take(dataflow::importCanonicalDataflow(source.dataflow, artifacts));
  auto context = makeContext();
  const DeadlineControl spatialPnrDeadline{std::chrono::steady_clock::now() +
                                           kSpatialPnrLimit};
  const loom::ExecutionControlView spatialPnrExecution =
      spatialPnrDeadline.view();
  auto hardware = loom::eda::test::buildMappedBuiltinSpatialMappingFixture(
      "cgra-budget-profile", dataflow, targetScale, context, spatialPnrConfig,
      spatialPnrExecution, artifacts, blobs,
      loom::eda::test::MappedRtlRouteCoverage::AnyLegal);
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
      take(loom::evaluation::models::evaluateCgraSimulationWithDiagnostics(
          prepared, {loom::runtime::gem5MaximumSpatialWork, warmupDeadline},
          artifacts, blobs));
  std::optional<loom::ArtifactRootReference> preRepairEvidence;
  std::optional<loom::ArtifactRootReference> repairConstraint;
  std::optional<loom::ArtifactRootReference> parentSystemMapping;
  std::optional<loom::ArtifactRootReference> repairedSpatialMapping;
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
      if (!repaired.spatialMapping)
        continue;
      auto candidateSpatial = std::move(*repaired.spatialMapping);
      auto candidatePrepared =
          take(loom::evaluation::models::prepareCgraSimulationEvaluation(
              source.dataflow, hardware.module.reference(),
              candidateSpatial.reference(), source.workload,
              source.runtimeInput, resolvedConfig, artifacts, blobs));
      const auto candidateDeadline =
          std::chrono::steady_clock::now() + kQualificationLimit;
      auto candidateWarmup =
          take(loom::evaluation::models::evaluateCgraSimulationWithDiagnostics(
              candidatePrepared,
              {loom::runtime::gem5MaximumSpatialWork, candidateDeadline},
              artifacts, blobs));
      if (!completed(candidateWarmup))
        continue;
      repairConstraint = repaired.constraintSet;
      repairedSpatialMapping = candidateSpatial.reference();
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
        take(loom::evaluation::models::evaluateCgraSimulationWithDiagnostics(
            prepared, {loom::runtime::gem5MaximumSpatialWork, deadline},
            artifacts, blobs));
    const auto evidence = take(loom::evaluation::publishEvaluationEvidence(
        evaluated.evidence, artifacts));
    measurements.push_back(measurementJson(evaluated, evidence));
  }

  llvm::json::Object report{
      {"schema", "loom.cgra_budget_profile.2"},
      {"workload", argv[3]},
      {"operator_id", argv[4]},
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
      {"initial_spatial_mapping", referenceJson(initialSpatialMapping)},
      {"spatial_mapping", referenceJson(hardware.spatialMapping.reference())},
      {"repaired_spatial_mapping",
       repairedSpatialMapping
           ? llvm::json::Value(referenceJson(*repairedSpatialMapping))
           : llvm::json::Value(nullptr)},
      {"parent_system_mapping",
       parentSystemMapping
           ? llvm::json::Value(referenceJson(*parentSystemMapping))
           : llvm::json::Value(nullptr)},
      {"transport_repair_constraint",
       repairConstraint ? llvm::json::Value(referenceJson(*repairConstraint))
                        : llvm::json::Value(nullptr)},
      {"pre_repair_evidence",
       preRepairEvidence ? llvm::json::Value(referenceJson(*preRepairEvidence))
                         : llvm::json::Value(nullptr)},
      {"warmup_evidence", referenceJson(warmupEvidence)},
      {"measurements", std::move(measurements)},
  };
  llvm::outs() << llvm::formatv("{0:2}\n",
                                llvm::json::Value(std::move(report)));
  return EXIT_SUCCESS;
}
