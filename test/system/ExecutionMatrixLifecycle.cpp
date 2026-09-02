#include "ExecutionMatrixLifecycle.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Deployment/DeploymentDiagnostics.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "EDA/Adapters/OpenSource/MappedRtlExecution.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Runtime/Gem5SystemExecution.h"

#include "MappedRtlSimulationTestSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if !defined(LOOM_TEST_BUILD_JOBS) || !defined(LOOM_TEST_RTL_BUILD_WORKER_LIMIT)
#error "system execution test build limits must be configured"
#endif

namespace loom::system_test {
namespace {

inline constexpr std::size_t artifactImportEntryLimit = 256;
inline constexpr std::size_t fabricImportEntryLimit = 64;
inline constexpr std::size_t systemMappingImportEntryLimit = 64;
inline constexpr std::size_t configurationProjectionEntryLimit = 64;
inline constexpr std::size_t gem5FactsEntryLimit = 8;

std::optional<std::uint64_t> processCpuNanoseconds() {
  timespec current{};
  if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current) != 0 ||
      current.tv_sec < 0 || current.tv_nsec < 0 ||
      current.tv_nsec >= 1'000'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t seconds = current.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() -
                 static_cast<std::uint64_t>(current.tv_nsec)) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + current.tv_nsec;
}

std::optional<std::uint64_t> timevalNanoseconds(const timeval &value) {
  if (value.tv_sec < 0 || value.tv_usec < 0 || value.tv_usec >= 1'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t subsecond =
      static_cast<std::uint64_t>(value.tv_usec) * 1000;
  const std::uint64_t seconds = value.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() - subsecond) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + subsecond;
}

/// The CLOCK_MONOTONIC reading that names an interval's start boundary. It is
/// the alignment key for external samplers recording on the same clock.
std::optional<std::uint64_t> monotonicNanoseconds() {
  timespec current{};
  if (::clock_gettime(CLOCK_MONOTONIC, &current) != 0 || current.tv_sec < 0 ||
      current.tv_nsec < 0 || current.tv_nsec >= 1'000'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t seconds = current.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() -
                 static_cast<std::uint64_t>(current.tv_nsec)) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + current.tv_nsec;
}

struct ResourceSnapshot final {
  std::chrono::steady_clock::time_point wall;
  std::optional<std::uint64_t> monotonicNanoseconds;
  std::optional<std::uint64_t> selfCpuNanoseconds;
  std::optional<std::uint64_t> childCpuNanoseconds;
  std::optional<std::uint64_t> selfProcessLifetimeHighWaterRssKib;
  std::optional<std::uint64_t> maximumWaitedDescendantProcessRssKib;
};

ResourceSnapshot captureResources() {
  ResourceSnapshot snapshot;
  snapshot.wall = std::chrono::steady_clock::now();
  snapshot.monotonicNanoseconds = monotonicNanoseconds();
  snapshot.selfCpuNanoseconds = processCpuNanoseconds();
  rusage selfUsage{};
  if (::getrusage(RUSAGE_SELF, &selfUsage) == 0 && selfUsage.ru_maxrss >= 0)
    snapshot.selfProcessLifetimeHighWaterRssKib = selfUsage.ru_maxrss;
  rusage usage{};
  if (::getrusage(RUSAGE_CHILDREN, &usage) == 0) {
    auto user = timevalNanoseconds(usage.ru_utime);
    auto system = timevalNanoseconds(usage.ru_stime);
    if (user && system &&
        *system <= std::numeric_limits<std::uint64_t>::max() - *user)
      snapshot.childCpuNanoseconds = *user + *system;
    if (usage.ru_maxrss >= 0)
      snapshot.maximumWaitedDescendantProcessRssKib = usage.ru_maxrss;
  }
  return snapshot;
}

std::optional<std::uint64_t> difference(std::optional<std::uint64_t> end,
                                        std::optional<std::uint64_t> begin) {
  if (!end || !begin || *end < *begin)
    return std::nullopt;
  return *end - *begin;
}

llvm::StringRef spelling(ExecutionMatrixLifecycleOperation operation) {
  switch (operation) {
  case ExecutionMatrixLifecycleOperation::Setup:
    return "setup";
  case ExecutionMatrixLifecycleOperation::DataflowConstructionAndPublication:
    return "dataflow_construction_and_publication";
  case ExecutionMatrixLifecycleOperation::
      FabricModuleConstructionAndFinalization:
    return "fabric_module_construction_and_finalization";
  case ExecutionMatrixLifecycleOperation::TechMapping:
    return "tech_mapping";
  case ExecutionMatrixLifecycleOperation::SpatialPnr:
    return "spatial_pnr";
  case ExecutionMatrixLifecycleOperation::
      SystemFabricAndInterconnectConstruction:
    return "system_fabric_and_interconnect_construction";
  case ExecutionMatrixLifecycleOperation::
      ConfigurationAbiAndHardwareImplementationGeneration:
    return "configuration_abi_and_hardware_implementation_generation";
  case ExecutionMatrixLifecycleOperation::SystemMappingAndPnr:
    return "system_mapping_and_pnr";
  case ExecutionMatrixLifecycleOperation::GuestCompileAndLink:
    return "guest_compile_and_link";
  case ExecutionMatrixLifecycleOperation::
      RuntimeBindingAndDeploymentFinalization:
    return "runtime_binding_and_deployment_finalization";
  case ExecutionMatrixLifecycleOperation::WorkloadAndRuntimeInputPublication:
    return "workload_and_runtime_input_publication";
  case ExecutionMatrixLifecycleOperation::HostLifecycle:
    return "host_lifecycle";
  case ExecutionMatrixLifecycleOperation::Gem5Readiness:
    return "gem5_readiness";
  case ExecutionMatrixLifecycleOperation::Gem5Binding:
    return "gem5_binding";
  case ExecutionMatrixLifecycleOperation::RequestConstruction:
    return "request_construction";
  case ExecutionMatrixLifecycleOperation::OrdinaryPrepare:
    return "ordinary_prepare";
  case ExecutionMatrixLifecycleOperation::OrdinaryExternalExecution:
    return "ordinary_external_execution";
  case ExecutionMatrixLifecycleOperation::OrdinaryImportAndEvidenceAssembly:
    return "ordinary_import_and_evidence_assembly";
  case ExecutionMatrixLifecycleOperation::OrdinaryEvidencePublication:
    return "ordinary_evidence_publication";
  case ExecutionMatrixLifecycleOperation::OrdinaryExecutionImport:
    return "ordinary_execution_import";
  case ExecutionMatrixLifecycleOperation::DiagnosticPrepare:
    return "diagnostic_prepare";
  case ExecutionMatrixLifecycleOperation::DiagnosticExternalExecution:
    return "diagnostic_external_execution";
  case ExecutionMatrixLifecycleOperation::DiagnosticImportAndEvidenceAssembly:
    return "diagnostic_import_and_evidence_assembly";
  case ExecutionMatrixLifecycleOperation::DiagnosticEvidencePublication:
    return "diagnostic_evidence_publication";
  case ExecutionMatrixLifecycleOperation::DiagnosticExecutionImport:
    return "diagnostic_execution_import";
  case ExecutionMatrixLifecycleOperation::OracleVerification:
    return "oracle_verification";
  case ExecutionMatrixLifecycleOperation::Cleanup:
    return "cleanup";
  }
  llvm_unreachable("unknown execution-matrix lifecycle operation");
}

void printOptional(llvm::raw_ostream &output,
                   std::optional<std::uint64_t> value) {
  if (value)
    output << *value;
  else
    output << "unavailable";
}

void emitInvocationKey(llvm::raw_ostream &output,
                       const ExecutionMatrixInvocation &invocation) {
  output << " cell=" << executionMatrixCellName(invocation.cell)
         << " attempt=" << executionMatrixAttemptName(invocation.attempt)
         << " invocation=" << executionMatrixInvocationName(invocation);
}

llvm::StringRef lifecycleParent(ExecutionMatrixLifecycleOperation operation) {
  switch (operation) {
  case ExecutionMatrixLifecycleOperation::DataflowConstructionAndPublication:
  case ExecutionMatrixLifecycleOperation::
      FabricModuleConstructionAndFinalization:
  case ExecutionMatrixLifecycleOperation::TechMapping:
  case ExecutionMatrixLifecycleOperation::SpatialPnr:
  case ExecutionMatrixLifecycleOperation::
      SystemFabricAndInterconnectConstruction:
  case ExecutionMatrixLifecycleOperation::
      ConfigurationAbiAndHardwareImplementationGeneration:
  case ExecutionMatrixLifecycleOperation::SystemMappingAndPnr:
  case ExecutionMatrixLifecycleOperation::GuestCompileAndLink:
  case ExecutionMatrixLifecycleOperation::
      RuntimeBindingAndDeploymentFinalization:
  case ExecutionMatrixLifecycleOperation::WorkloadAndRuntimeInputPublication:
    return "setup";
  case ExecutionMatrixLifecycleOperation::Gem5Readiness:
  case ExecutionMatrixLifecycleOperation::Gem5Binding:
  case ExecutionMatrixLifecycleOperation::RequestConstruction:
  case ExecutionMatrixLifecycleOperation::OrdinaryPrepare:
  case ExecutionMatrixLifecycleOperation::OrdinaryExternalExecution:
  case ExecutionMatrixLifecycleOperation::OrdinaryImportAndEvidenceAssembly:
  case ExecutionMatrixLifecycleOperation::OrdinaryEvidencePublication:
  case ExecutionMatrixLifecycleOperation::OrdinaryExecutionImport:
  case ExecutionMatrixLifecycleOperation::DiagnosticPrepare:
  case ExecutionMatrixLifecycleOperation::DiagnosticExternalExecution:
  case ExecutionMatrixLifecycleOperation::DiagnosticImportAndEvidenceAssembly:
  case ExecutionMatrixLifecycleOperation::DiagnosticEvidencePublication:
  case ExecutionMatrixLifecycleOperation::DiagnosticExecutionImport:
    return "host_lifecycle";
  case ExecutionMatrixLifecycleOperation::Setup:
  case ExecutionMatrixLifecycleOperation::HostLifecycle:
  case ExecutionMatrixLifecycleOperation::OracleVerification:
  case ExecutionMatrixLifecycleOperation::Cleanup:
    return "none";
  }
  llvm_unreachable("unknown execution-matrix lifecycle operation");
}

llvm::StringRef
attemptPairLifecycleParent(ExecutionMatrixLifecycleOperation operation) {
  if (operation == ExecutionMatrixLifecycleOperation::Gem5Readiness ||
      operation == ExecutionMatrixLifecycleOperation::Gem5Binding)
    return "none";
  return lifecycleParent(operation);
}

struct CacheRow final {
  llvm::StringRef cache;
  llvm::StringRef hitValidation;
  std::uint64_t requests = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t constructionAttempts = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t uncachedConstructions = 0;
  std::uint64_t unsupportedConstructions = 0;
  std::uint64_t failedConstructions = 0;
  std::uint64_t revalidationCount = 0;
  std::uint64_t revalidatedArtifactBytes = 0;
  std::uint64_t revalidatedBlobBytes = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t minimumRetainedBytes = 0;
  std::uint64_t entryCount = 0;
};

void emitCacheRow(const ExecutionMatrixInvocation &invocation,
                  const CacheRow &row) {
  llvm::outs() << "execution-matrix-cache"
               << " schema=loom.execution_matrix_cache.2";
  emitInvocationKey(llvm::outs(), invocation);
  llvm::outs() << " cache=" << row.cache
               << " hit_validation=" << row.hitValidation
               << " requests=" << row.requests << " hits=" << row.hits
               << " misses=" << row.misses
               << " construction_attempts=" << row.constructionAttempts
               << " unique_constructions=" << row.uniqueConstructions
               << " uncached_constructions=" << row.uncachedConstructions
               << " unsupported_constructions=" << row.unsupportedConstructions
               << " failed_constructions=" << row.failedConstructions
               << " revalidation_count=" << row.revalidationCount
               << " revalidated_artifact_bytes=" << row.revalidatedArtifactBytes
               << " revalidated_blob_bytes=" << row.revalidatedBlobBytes
               << " construction_wall_ns=" << row.constructionNanoseconds
               << " minimum_retained_bytes=" << row.minimumRetainedBytes
               << " entries=" << row.entryCount << '\n';
}

void emitFactsOperationRow(
    const ExecutionMatrixInvocation &invocation, llvm::StringRef operation,
    const runtime::Gem5SystemFactsOperationStatistics &statistics) {
  llvm::outs() << "execution-matrix-facts-operation"
               << " schema=loom.execution_matrix_facts_operation.2";
  emitInvocationKey(llvm::outs(), invocation);
  llvm::outs() << " interval_kind=inclusive parent="
               << (operation == "derive_facts" ? "none" : "derive_facts")
               << " operation=" << operation
               << " invocations=" << statistics.invocations
               << " wall_ns=" << statistics.wallNanoseconds << " self_cpu_ns=";
  printOptional(
      llvm::outs(),
      statistics.selfCpuObservationCount == statistics.invocations
          ? std::optional<std::uint64_t>(statistics.selfCpuNanoseconds)
          : std::nullopt);
  llvm::outs() << " child_cpu_ns=";
  printOptional(
      llvm::outs(),
      statistics.childCpuObservationCount == statistics.invocations
          ? std::optional<std::uint64_t>(statistics.childCpuNanoseconds)
          : std::nullopt);
  llvm::outs() << '\n';
}

struct DeploymentOperationRow final {
  deployment::DeploymentConstructionMode mode;
  deployment::DeploymentConstructionOperation operation;
  std::uint64_t invocations = 0;
  std::uint64_t wallNanoseconds = 0;
  std::optional<std::uint64_t> selfCpuNanoseconds = 0;
  std::optional<std::uint64_t> childCpuNanoseconds = 0;
};

void addSaturated(std::uint64_t &destination, std::uint64_t value) {
  destination = value > std::numeric_limits<std::uint64_t>::max() - destination
                    ? std::numeric_limits<std::uint64_t>::max()
                    : destination + value;
}

void emitDeploymentOperationRows(
    const ExecutionMatrixInvocation &invocation,
    llvm::ArrayRef<deployment::DeploymentConstructionOperationStatistics>
        observations) {
  std::vector<DeploymentOperationRow> rows;
  for (const auto &observation : observations) {
    auto row =
        llvm::find_if(rows, [&](const DeploymentOperationRow &candidate) {
          return candidate.mode == observation.mode &&
                 candidate.operation == observation.operation;
        });
    if (row == rows.end()) {
      rows.push_back({observation.mode, observation.operation});
      row = std::prev(rows.end());
    }
    addSaturated(row->invocations, 1);
    addSaturated(row->wallNanoseconds, observation.durationNanoseconds);
    if (row->selfCpuNanoseconds && observation.selfCpuNanoseconds)
      addSaturated(*row->selfCpuNanoseconds, *observation.selfCpuNanoseconds);
    else
      row->selfCpuNanoseconds = std::nullopt;
    if (row->childCpuNanoseconds && observation.childCpuNanoseconds)
      addSaturated(*row->childCpuNanoseconds, *observation.childCpuNanoseconds);
    else
      row->childCpuNanoseconds = std::nullopt;
  }
  for (const DeploymentOperationRow &row : rows) {
    llvm::outs() << "execution-matrix-deployment-operation"
                 << " schema=loom.execution_matrix_deployment_operation.2";
    emitInvocationKey(llvm::outs(), invocation);
    llvm::outs() << " interval_kind=exclusive parent=none"
                 << " mode="
                 << deployment::deploymentConstructionModeName(row.mode)
                 << " operation="
                 << deployment::deploymentConstructionOperationName(
                        row.operation)
                 << " invocations=" << row.invocations
                 << " wall_ns=" << row.wallNanoseconds << " self_cpu_ns=";
    printOptional(llvm::outs(), row.selfCpuNanoseconds);
    llvm::outs() << " child_cpu_ns=";
    printOptional(llvm::outs(), row.childCpuNanoseconds);
    llvm::outs() << '\n';
  }
}

llvm::StringRef systemRtlCommandRole(std::size_t commandOrdinal) {
  if (commandOrdinal < systemRtlSpatialLaunchCount)
    return "rtl_verilation";
  if (commandOrdinal < 2 * systemRtlSpatialLaunchCount)
    return "rtl_build";
  return "rtl_controller";
}

ExecutionMatrixLifecycleOperation setupLifecycleOperation(
    eda::test::MappedSpatialHardwareFixtureOperation operation) {
  using HardwareOperation = eda::test::MappedSpatialHardwareFixtureOperation;
  switch (operation) {
  case HardwareOperation::DataflowPublication:
    return ExecutionMatrixLifecycleOperation::
        DataflowConstructionAndPublication;
  case HardwareOperation::FabricModuleConstructionAndFinalization:
    return ExecutionMatrixLifecycleOperation::
        FabricModuleConstructionAndFinalization;
  case HardwareOperation::TechMapping:
    return ExecutionMatrixLifecycleOperation::TechMapping;
  case HardwareOperation::SpatialPnr:
    return ExecutionMatrixLifecycleOperation::SpatialPnr;
  case HardwareOperation::SystemFabricAndInterconnectConstruction:
    return ExecutionMatrixLifecycleOperation::
        SystemFabricAndInterconnectConstruction;
  case HardwareOperation::ConfigurationAbiAndHardwareImplementationGeneration:
    return ExecutionMatrixLifecycleOperation::
        ConfigurationAbiAndHardwareImplementationGeneration;
  }
  llvm_unreachable("unknown mapped hardware fixture operation");
}

} // namespace

llvm::Error emitExecutionMatrixRunSummary(
    const ExecutionMatrixInvocation &invocation,
    std::uint64_t deterministicWork,
    std::uint64_t maximumWaitedDescendantProcessRssKib,
    const runtime::Gem5SystemAttemptProfile *profile,
    llvm::ArrayRef<runtime::Gem5SpatialInvocationProjection>
        spatialInvocations) {
  if (!profile && !spatialInvocations.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "ordinary execution has diagnostic invocation projections");
  llvm::outs() << "execution-matrix"
               << " schema=loom.execution_matrix_summary.4";
  emitInvocationKey(llvm::outs(), invocation);
  llvm::outs() << " deterministic_work=" << deterministicWork
               << " maximum_waited_descendant_process_rss_kib="
               << maximumWaitedDescendantProcessRssKib
               << " rss_scope=process_lifetime";
  if (profile) {
    std::uint64_t acceleratorReferenceCycles = 0;
    std::uint64_t unavailableAcceleratorCycleCount = 0;
    for (const runtime::Gem5SpatialInvocationProjection &spatial :
         spatialInvocations) {
      if (!spatial.acceleratorReferenceCycles) {
        ++unavailableAcceleratorCycleCount;
        continue;
      }
      if (*spatial.acceleratorReferenceCycles >
          std::numeric_limits<std::uint64_t>::max() -
              acceleratorReferenceCycles)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "accelerator cycle report overflows its domain");
      acceleratorReferenceCycles += *spatial.acceleratorReferenceCycles;
    }
    llvm::outs() << " gem5_configuration_wall_us="
                 << profile->configurationWallNanoseconds / 1000
                 << " gem5_simulation_wall_us="
                 << profile->simulationWallNanoseconds / 1000
                 << " gem5_simulation_cpu_us="
                 << profile->gem5SimulationProcessCpuNanoseconds / 1000
                 << " observation_wall_us="
                 << profile->observationWallNanoseconds / 1000
                 << " observation_cpu_us="
                 << profile->observationProcessCpuNanoseconds / 1000
                 << " bridge_callback_cpu_us="
                 << profile->bridgeCallbackCpuNanoseconds / 1000
                 << " bridge_wait_us="
                 << profile->bridgeEngineWaitNanoseconds / 1000
                 << " bridge_messages=" << profile->bridgeMessageCount
                 << " accelerator_invocations="
                 << profile->acceleratorInvocationCount
                 << " accelerator_cycles=" << acceleratorReferenceCycles
                 << " accelerator_cycle_unavailable_count="
                 << unavailableAcceleratorCycleCount;
    if (profile->managedEngineStartup)
      llvm::outs() << " managed_engine_startup_wall_us="
                   << profile->managedEngineStartup->wallNanoseconds / 1000
                   << " managed_engine_startup_self_cpu_us="
                   << profile->managedEngineStartup->selfProcessCpuNanoseconds /
                          1000;
    if (profile->externalEngineSocketReadiness)
      llvm::outs()
          << " external_engine_socket_readiness_wall_us="
          << profile->externalEngineSocketReadiness->wallNanoseconds / 1000
          << " external_engine_socket_readiness_self_cpu_us="
          << profile->externalEngineSocketReadiness->selfProcessCpuNanoseconds /
                 1000;
    if (profile->engineProcessCpuNanoseconds)
      llvm::outs() << " engine_process_cpu_us="
                   << *profile->engineProcessCpuNanoseconds / 1000;
    if (profile->cgraEngine)
      llvm::outs() << " cgra_engine_active_wall_us="
                   << profile->cgraEngine->activeWallNanoseconds / 1000
                   << " cgra_engine_active_cpu_us="
                   << profile->cgraEngine->activeProcessCpuNanoseconds / 1000
                   << " cgra_event_frames="
                   << profile->cgraEngine->eventFrameCount;
  }
  llvm::outs() << '\n';
  return llvm::Error::success();
}

class ExecutionMatrixLifecycleRecorder::Impl final {
public:
  struct Record final {
    std::uint64_t sequence = 0;
    ExecutionMatrixLifecycleOperation operation;
    std::uint64_t wallNanoseconds = 0;
    std::optional<std::uint64_t> selfCpuNanoseconds;
    std::optional<std::uint64_t> childCpuNanoseconds;
    std::optional<std::uint64_t> selfProcessLifetimeHighWaterRssKib;
    std::optional<std::uint64_t> maximumWaitedDescendantProcessRssKib;
    std::optional<std::uint64_t> beginMonotonicNanoseconds;
  };

  std::uint64_t reserveSequence() { return nextSequence_++; }

  std::uint64_t record(std::uint64_t sequence,
                       ExecutionMatrixLifecycleOperation operation,
                       const ResourceSnapshot &begin,
                       const ResourceSnapshot &end) {
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end.wall - begin.wall);
    const std::uint64_t wallNanoseconds =
        static_cast<std::uint64_t>(std::max<std::int64_t>(0, wall.count()));
    records_.push_back(
        {sequence, operation, wallNanoseconds,
         difference(end.selfCpuNanoseconds, begin.selfCpuNanoseconds),
         difference(end.childCpuNanoseconds, begin.childCpuNanoseconds),
         end.selfProcessLifetimeHighWaterRssKib,
         end.maximumWaitedDescendantProcessRssKib, begin.monotonicNanoseconds});
    return wallNanoseconds;
  }

  static void emitObservations(llvm::raw_ostream &output,
                               const Record &record) {
    output << " operation=" << spelling(record.operation)
           << " wall_ns=" << record.wallNanoseconds << " self_cpu_ns=";
    printOptional(output, record.selfCpuNanoseconds);
    output << " child_cpu_ns=";
    printOptional(output, record.childCpuNanoseconds);
    output << " self_process_lifetime_high_water_rss_kib_snapshot=";
    printOptional(output, record.selfProcessLifetimeHighWaterRssKib);
    output << " maximum_waited_descendant_process_rss_kib_snapshot=";
    printOptional(output, record.maximumWaitedDescendantProcessRssKib);
    output << " begin_monotonic_ns=";
    printOptional(output, record.beginMonotonicNanoseconds);
    output << '\n';
  }

  void emit(const ExecutionMatrixInvocation &invocation) const {
    std::vector<Record> ordered = records_;
    llvm::sort(ordered, [](const Record &lhs, const Record &rhs) {
      return lhs.sequence < rhs.sequence;
    });
    for (const Record &record : ordered) {
      llvm::outs() << "execution-matrix-lifecycle"
                   << " schema=loom.execution_matrix_lifecycle.4.1";
      emitInvocationKey(llvm::outs(), invocation);
      llvm::outs() << " interval_kind=inclusive parent="
                   << lifecycleParent(record.operation);
      emitObservations(llvm::outs(), record);
    }
  }

  void emitAttemptPair(ExecutionMatrixCell cell) const {
    std::vector<Record> ordered = records_;
    llvm::sort(ordered, [](const Record &lhs, const Record &rhs) {
      return lhs.sequence < rhs.sequence;
    });
    for (const Record &record : ordered) {
      llvm::outs() << "execution-matrix-attempt-pair-lifecycle"
                   << " schema=loom.execution_matrix_attempt_pair_lifecycle.2"
                   << " cell=" << executionMatrixCellName(cell)
                   << " scope=attempt_pair"
                   << " interval_kind=inclusive parent="
                   << attemptPairLifecycleParent(record.operation);
      emitObservations(llvm::outs(), record);
    }
  }

  std::uint64_t
  operationCount(ExecutionMatrixLifecycleOperation operation) const {
    return static_cast<std::uint64_t>(
        llvm::count_if(records_, [&](const Record &record) {
          return record.operation == operation;
        }));
  }

private:
  std::uint64_t nextSequence_ = 0;
  std::vector<Record> records_;
};

class ExecutionMatrixLifecycleTimer::Impl final {
public:
  Impl(ExecutionMatrixLifecycleRecorder &recorder,
       ExecutionMatrixLifecycleOperation operation, std::uint64_t sequence)
      : recorder(&recorder), operation(operation), sequence(sequence),
        begin(captureResources()) {}

  ExecutionMatrixLifecycleRecorder *recorder;
  ExecutionMatrixLifecycleOperation operation;
  std::uint64_t sequence;
  ResourceSnapshot begin;
};

ExecutionMatrixLifecycleRecorder::ExecutionMatrixLifecycleRecorder()
    : impl_(std::make_unique<Impl>()) {}
ExecutionMatrixLifecycleRecorder::~ExecutionMatrixLifecycleRecorder() = default;

void ExecutionMatrixLifecycleRecorder::emit(
    const ExecutionMatrixInvocation &invocation) const {
  impl_->emit(invocation);
}

void ExecutionMatrixLifecycleRecorder::emitAttemptPair(
    ExecutionMatrixCell cell) const {
  impl_->emitAttemptPair(cell);
}

std::uint64_t ExecutionMatrixLifecycleRecorder::operationCount(
    ExecutionMatrixLifecycleOperation operation) const {
  return impl_->operationCount(operation);
}

std::uint64_t fullBudgetRtlBuildJobs() {
  std::uint64_t jobs = 1;
  for (std::uint64_t candidate = 1; candidate <= LOOM_TEST_BUILD_JOBS;
       ++candidate)
    if (eda::open_source::isMappedRtlParallelismCount(candidate))
      jobs = candidate;
  return jobs;
}

void emitExecutionMatrixExternalCommands(
    ExecutionMatrixInvocation invocation,
    llvm::ArrayRef<external_tool::ExternalToolCommandExecutionObservation>
        commands) {
  if (invocation.cell != ExecutionMatrixCell::SystemRtl)
    return;
  if (commands.size() != systemRtlCommandCount)
    llvm::report_fatal_error("System RTL execution did not report one "
                             "Verilation and one build per spatial launch "
                             "and one controller");
  for (const auto indexed : llvm::enumerate(commands)) {
    const external_tool::ExternalToolCommandExecutionObservation &command =
        indexed.value();
    if (command.commandOrdinal != indexed.index() || command.exitCode != 0)
      llvm::report_fatal_error(
          "System RTL command observations are not canonical successes");
    llvm::outs() << "execution-matrix-external-command"
                 << " schema=loom.execution_matrix_external_command.3";
    emitInvocationKey(llvm::outs(), invocation);
    llvm::outs() << " command_ordinal=" << command.commandOrdinal
                 << " command_role=" << systemRtlCommandRole(indexed.index())
                 << " wall_ns=" << command.wallNanoseconds
                 << " exit_code=" << command.exitCode
                 << " total_build_jobs=" << fullBudgetRtlBuildJobs()
                 << " build_worker_limit=" << LOOM_TEST_RTL_BUILD_WORKER_LIMIT
                 << '\n';
  }
}

ExecutionMatrixLifecycleTimer::ExecutionMatrixLifecycleTimer(
    ExecutionMatrixLifecycleRecorder &recorder,
    ExecutionMatrixLifecycleOperation operation)
    : impl_(std::make_unique<Impl>(recorder, operation,
                                   recorder.impl_->reserveSequence())) {}

ExecutionMatrixLifecycleTimer::ExecutionMatrixLifecycleTimer(
    ExecutionMatrixLifecycleRecorder &recorder,
    eda::test::MappedSpatialHardwareFixtureOperation operation)
    : ExecutionMatrixLifecycleTimer(recorder,
                                    setupLifecycleOperation(operation)) {}

ExecutionMatrixLifecycleTimer::~ExecutionMatrixLifecycleTimer() {
  if (impl_)
    (void)finish();
}

std::uint64_t ExecutionMatrixLifecycleTimer::finish() {
  if (!impl_)
    llvm_unreachable("execution-matrix lifecycle timer finished twice");
  const ResourceSnapshot end = captureResources();
  const std::uint64_t wallNanoseconds = impl_->recorder->impl_->record(
      impl_->sequence, impl_->operation, impl_->begin, end);
  impl_.reset();
  return wallNanoseconds;
}

namespace {

struct ImportStatisticsSnapshot final {
  std::vector<deployment::DeploymentConstructionOperationStatistics> deployment;
  evaluation::ArtifactImportCacheStatistics artifact;
  fabric::FabricArtifactImportSessionStatistics fabric;
  hardware::ConfigurationABIImportSessionStatistics configurationAbi;
  mapping::SystemMappingImportSessionStatistics systemMapping;
  deployment::ConfigurationImageProjectionSessionStatistics
      configurationProjection;
  runtime::Gem5SystemFactsSessionStatistics gem5Facts;
};

std::uint64_t counterDelta(std::uint64_t current, std::uint64_t baseline) {
  if (current < baseline)
    llvm_unreachable("execution-matrix import statistic regressed");
  return current - baseline;
}

CacheRow cacheRow(const evaluation::ArtifactImportCacheStatistics &current,
                  const evaluation::ArtifactImportCacheStatistics *baseline) {
  const evaluation::ArtifactImportCacheStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  return {
      "artifact_import",
      "artifact_revalidation",
      counterDelta(current.importRequests, prior.importRequests),
      counterDelta(current.cacheHits, prior.cacheHits),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.uniqueConstructions, prior.uniqueConstructions),
      counterDelta(current.uncachedConstructions, prior.uncachedConstructions),
      0,
      0,
      counterDelta(current.revalidationCount, prior.revalidationCount),
      counterDelta(current.revalidatedBytes, prior.revalidatedBytes),
      0,
      counterDelta(current.constructionNanoseconds,
                   prior.constructionNanoseconds),
      current.minimumRetainedBytes,
      current.entryCount};
}

CacheRow
cacheRow(const fabric::FabricArtifactImportSessionStatistics &current,
         const fabric::FabricArtifactImportSessionStatistics *baseline) {
  const fabric::FabricArtifactImportSessionStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  return {
      "fabric_import",
      "artifact_revalidation",
      counterDelta(current.importRequests, prior.importRequests),
      counterDelta(current.cacheHits, prior.cacheHits),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.uniqueConstructions, prior.uniqueConstructions),
      counterDelta(current.uncachedConstructions, prior.uncachedConstructions),
      0,
      0,
      counterDelta(current.revalidationCount, prior.revalidationCount),
      counterDelta(current.revalidatedBytes, prior.revalidatedBytes),
      0,
      counterDelta(current.constructionNanoseconds,
                   prior.constructionNanoseconds),
      current.retainedPayloadBytes,
      current.entryCount};
}

CacheRow
cacheRow(const hardware::ConfigurationABIImportSessionStatistics &current,
         const hardware::ConfigurationABIImportSessionStatistics *baseline) {
  const hardware::ConfigurationABIImportSessionStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  return {"configuration_abi_import",
          "immutable_session_domain",
          counterDelta(current.importRequests, prior.importRequests),
          counterDelta(current.cacheHits, prior.cacheHits),
          counterDelta(current.cacheMisses, prior.cacheMisses),
          counterDelta(current.cacheMisses, prior.cacheMisses),
          counterDelta(current.uniqueConstructions, prior.uniqueConstructions),
          0,
          0,
          0,
          0,
          0,
          0,
          counterDelta(current.constructionNanoseconds,
                       prior.constructionNanoseconds),
          current.retainedBytes,
          current.entryCount};
}

CacheRow
cacheRow(const mapping::SystemMappingImportSessionStatistics &current,
         const mapping::SystemMappingImportSessionStatistics *baseline) {
  const mapping::SystemMappingImportSessionStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  return {
      "system_mapping_import",
      "immutable_session_domain",
      counterDelta(current.importRequests, prior.importRequests),
      counterDelta(current.cacheHits, prior.cacheHits),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.uniqueConstructions, prior.uniqueConstructions),
      counterDelta(current.uncachedConstructions, prior.uncachedConstructions),
      0,
      0,
      0,
      0,
      0,
      counterDelta(current.constructionNanoseconds,
                   prior.constructionNanoseconds),
      current.retainedBytes,
      current.entryCount};
}

CacheRow cacheRow(
    const deployment::ConfigurationImageProjectionSessionStatistics &current,
    const deployment::ConfigurationImageProjectionSessionStatistics *baseline) {
  const deployment::ConfigurationImageProjectionSessionStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  return {
      "configuration_image_projection",
      "immutable_session_domain",
      counterDelta(current.requests, prior.requests),
      counterDelta(current.cacheHits, prior.cacheHits),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.uniqueConstructions, prior.uniqueConstructions),
      counterDelta(current.uncachedConstructions, prior.uncachedConstructions),
      0,
      0,
      0,
      0,
      0,
      counterDelta(current.constructionNanoseconds,
                   prior.constructionNanoseconds),
      current.retainedBytes,
      current.entryCount};
}

CacheRow cacheRow(const runtime::Gem5SystemFactsSessionStatistics &current,
                  const runtime::Gem5SystemFactsSessionStatistics *baseline) {
  const runtime::Gem5SystemFactsSessionStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  return {
      "gem5_system_facts",
      "complete_closure_revalidation",
      counterDelta(current.requests, prior.requests),
      counterDelta(current.cacheHits, prior.cacheHits),
      counterDelta(current.cacheMisses, prior.cacheMisses),
      counterDelta(current.constructionAttempts, prior.constructionAttempts),
      counterDelta(current.uniqueConstructions, prior.uniqueConstructions),
      counterDelta(current.uncachedConstructions, prior.uncachedConstructions),
      counterDelta(current.unsupportedConstructions,
                   prior.unsupportedConstructions),
      counterDelta(current.failedConstructions, prior.failedConstructions),
      counterDelta(current.revalidationCount, prior.revalidationCount),
      counterDelta(current.revalidatedArtifactBytes,
                   prior.revalidatedArtifactBytes),
      counterDelta(current.revalidatedBlobBytes, prior.revalidatedBlobBytes),
      counterDelta(current.constructionNanoseconds,
                   prior.constructionNanoseconds),
      current.minimumRetainedBytes,
      current.entryCount};
}

CacheRow externalFileFingerprintRow(
    const runtime::Gem5SystemFactsSessionStatistics &current,
    const runtime::Gem5SystemFactsSessionStatistics *baseline) {
  const runtime::Gem5SystemFactsSessionStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  const std::uint64_t misses = counterDelta(
      current.externalFileFingerprintMisses, prior.externalFileFingerprintMisses);
  return {"gem5_external_file_fingerprint",
          "observed_file_identity",
          counterDelta(current.externalFileFingerprintRequests,
                       prior.externalFileFingerprintRequests),
          counterDelta(current.externalFileFingerprintHits,
                       prior.externalFileFingerprintHits),
          misses,
          misses,
          misses,
          0,
          0,
          0,
          0,
          0,
          0,
          counterDelta(current.externalFileFingerprintNanoseconds,
                       prior.externalFileFingerprintNanoseconds),
          current.externalFileFingerprintedBytes,
          current.externalFileFingerprintEntryCount};
}

runtime::Gem5SystemFactsOperationStatistics
operationDelta(const runtime::Gem5SystemFactsOperationStatistics &current,
               const runtime::Gem5SystemFactsOperationStatistics *baseline) {
  const runtime::Gem5SystemFactsOperationStatistics zero;
  const auto &prior = baseline ? *baseline : zero;
  return {counterDelta(current.invocations, prior.invocations),
          counterDelta(current.wallNanoseconds, prior.wallNanoseconds),
          counterDelta(current.selfCpuNanoseconds, prior.selfCpuNanoseconds),
          counterDelta(current.selfCpuObservationCount,
                       prior.selfCpuObservationCount),
          counterDelta(current.childCpuNanoseconds, prior.childCpuNanoseconds),
          counterDelta(current.childCpuObservationCount,
                       prior.childCpuObservationCount)};
}

bool exactBoundedCache(std::uint64_t requests, std::uint64_t hits,
                       std::uint64_t misses, std::uint64_t uniqueConstructions,
                       std::uint64_t uncachedConstructions,
                       std::uint64_t entryCount) {
  return requests == hits + misses && misses == uniqueConstructions &&
         uncachedConstructions == 0 && entryCount == uniqueConstructions &&
         hits != 0;
}

} // namespace

class ExecutionMatrixImportSessions::Impl final {
public:
  Impl(const ArtifactStore &artifacts, const BlobStore &blobs)
      : deploymentConstruction(),
        artifactImports(artifacts, &blobs, artifactImportEntryLimit),
        fabricImports(fabric::FabricArtifactImportSessionMode::Isolated,
                      fabricImportEntryLimit),
        configurationAbiImports(
            hardware::ConfigurationABIImportSessionMode::Isolated),
        systemMappingImports(artifacts, systemMappingImportEntryLimit),
        configurationProjections(artifacts, configurationProjectionEntryLimit),
        gem5Facts(artifacts, blobs,
                  runtime::Gem5SystemFactsSessionMode::Isolated,
                  gem5FactsEntryLimit) {}

  ImportStatisticsSnapshot statistics() const {
    return {deploymentConstruction.statistics(),
            artifactImports.statistics(),
            fabricImports.statistics(),
            configurationAbiImports.statistics(),
            systemMappingImports.statistics(),
            configurationProjections.statistics(),
            gem5Facts.statistics()};
  }

  deployment::DeploymentConstructionStatisticsSession deploymentConstruction;
  evaluation::ArtifactImportCacheScope artifactImports;
  fabric::FabricArtifactImportSession fabricImports;
  hardware::ConfigurationABIImportSession configurationAbiImports;
  mapping::SystemMappingImportSession systemMappingImports;
  deployment::ConfigurationImageProjectionSession configurationProjections;
  runtime::Gem5SystemFactsSession gem5Facts;
  std::optional<ImportStatisticsSnapshot> emittedStatistics;
};

ExecutionMatrixImportSessions::ExecutionMatrixImportSessions(
    const ArtifactStore &artifacts, const BlobStore &blobs)
    : impl_(std::make_unique<Impl>(artifacts, blobs)) {}
ExecutionMatrixImportSessions::~ExecutionMatrixImportSessions() = default;

ExecutionMatrixImportSummary ExecutionMatrixImportSessions::summary() const {
  const ImportStatisticsSnapshot statistics = impl_->statistics();
  const auto &artifact = statistics.artifact;
  const auto &fabric = statistics.fabric;
  const auto &abi = statistics.configurationAbi;
  const auto &mapping = statistics.systemMapping;
  const auto &projection = statistics.configurationProjection;
  const auto &facts = statistics.gem5Facts;
  return {facts.requests,
          facts.cacheHits,
          facts.cacheMisses,
          facts.constructionAttempts,
          facts.uniqueConstructions,
          facts.uncachedConstructions,
          facts.unsupportedConstructions,
          facts.failedConstructions,
          facts.revalidationCount,
          facts.revalidatedArtifactBytes,
          facts.revalidatedBlobBytes,
          facts.constructionNanosecondsSaved,
          facts.entryCount,
          artifact.cacheHits,
          fabric.cacheHits,
          abi.cacheHits,
          mapping.cacheHits,
          projection.cacheHits};
}

bool ExecutionMatrixImportSessions::reusedOneExactGem5FactsClosure() const {
  const ImportStatisticsSnapshot statistics = impl_->statistics();
  const auto &facts = statistics.gem5Facts;
  return facts.requests == 2 && facts.cacheHits == 1 &&
         facts.cacheMisses == 1 && facts.constructionAttempts == 1 &&
         facts.uniqueConstructions == 1 && facts.uncachedConstructions == 0 &&
         facts.unsupportedConstructions == 0 &&
         facts.failedConstructions == 0 && facts.revalidationCount == 1 &&
         facts.revalidatedArtifactBytes != 0 &&
         facts.revalidatedBlobBytes != 0 &&
         facts.constructionNanosecondsSaved != 0 && facts.entryCount == 1 &&
         facts.externalFileFingerprintRequests == 1 &&
         facts.externalFileFingerprintHits == 0 &&
         facts.externalFileFingerprintMisses == 1 &&
         facts.externalFileFingerprintedBytes != 0 &&
         facts.externalFileFingerprintEntryCount == 1 &&
         exactBoundedCache(statistics.artifact.importRequests,
                           statistics.artifact.cacheHits,
                           statistics.artifact.cacheMisses,
                           statistics.artifact.uniqueConstructions,
                           statistics.artifact.uncachedConstructions,
                           statistics.artifact.entryCount) &&
         exactBoundedCache(statistics.fabric.importRequests,
                           statistics.fabric.cacheHits,
                           statistics.fabric.cacheMisses,
                           statistics.fabric.uniqueConstructions,
                           statistics.fabric.uncachedConstructions,
                           statistics.fabric.entryCount) &&
         exactBoundedCache(statistics.configurationAbi.importRequests,
                           statistics.configurationAbi.cacheHits,
                           statistics.configurationAbi.cacheMisses,
                           statistics.configurationAbi.uniqueConstructions, 0,
                           statistics.configurationAbi.entryCount) &&
         exactBoundedCache(statistics.systemMapping.importRequests,
                           statistics.systemMapping.cacheHits,
                           statistics.systemMapping.cacheMisses,
                           statistics.systemMapping.uniqueConstructions,
                           statistics.systemMapping.uncachedConstructions,
                           statistics.systemMapping.entryCount) &&
         exactBoundedCache(
             statistics.configurationProjection.requests,
             statistics.configurationProjection.cacheHits,
             statistics.configurationProjection.cacheMisses,
             statistics.configurationProjection.uniqueConstructions,
             statistics.configurationProjection.uncachedConstructions,
             statistics.configurationProjection.entryCount);
}

bool ExecutionMatrixImportSessions::
    reusedOneExactGem5FactsClosureAcrossAttemptPair() const {
  const ImportStatisticsSnapshot statistics = impl_->statistics();
  const auto &facts = statistics.gem5Facts;
  return facts.requests == 4 && facts.cacheHits == 3 &&
         facts.cacheMisses == 1 && facts.constructionAttempts == 1 &&
         facts.uniqueConstructions == 1 && facts.uncachedConstructions == 0 &&
         facts.unsupportedConstructions == 0 &&
         facts.failedConstructions == 0 && facts.revalidationCount == 3 &&
         facts.revalidatedArtifactBytes != 0 &&
         facts.revalidatedBlobBytes != 0 &&
         facts.constructionNanosecondsSaved != 0 && facts.entryCount == 1 &&
         facts.externalFileFingerprintRequests == 2 &&
         facts.externalFileFingerprintHits == 1 &&
         facts.externalFileFingerprintMisses == 1 &&
         facts.externalFileFingerprintedBytes != 0 &&
         facts.externalFileFingerprintEntryCount == 1 &&
         exactBoundedCache(statistics.artifact.importRequests,
                           statistics.artifact.cacheHits,
                           statistics.artifact.cacheMisses,
                           statistics.artifact.uniqueConstructions,
                           statistics.artifact.uncachedConstructions,
                           statistics.artifact.entryCount) &&
         exactBoundedCache(statistics.fabric.importRequests,
                           statistics.fabric.cacheHits,
                           statistics.fabric.cacheMisses,
                           statistics.fabric.uniqueConstructions,
                           statistics.fabric.uncachedConstructions,
                           statistics.fabric.entryCount) &&
         exactBoundedCache(statistics.configurationAbi.importRequests,
                           statistics.configurationAbi.cacheHits,
                           statistics.configurationAbi.cacheMisses,
                           statistics.configurationAbi.uniqueConstructions, 0,
                           statistics.configurationAbi.entryCount) &&
         exactBoundedCache(statistics.systemMapping.importRequests,
                           statistics.systemMapping.cacheHits,
                           statistics.systemMapping.cacheMisses,
                           statistics.systemMapping.uniqueConstructions,
                           statistics.systemMapping.uncachedConstructions,
                           statistics.systemMapping.entryCount) &&
         exactBoundedCache(
             statistics.configurationProjection.requests,
             statistics.configurationProjection.cacheHits,
             statistics.configurationProjection.cacheMisses,
             statistics.configurationProjection.uniqueConstructions,
             statistics.configurationProjection.uncachedConstructions,
             statistics.configurationProjection.entryCount);
}

void ExecutionMatrixImportSessions::emitStatistics(
    const ExecutionMatrixInvocation &invocation) {
  ImportStatisticsSnapshot current = impl_->statistics();
  const ImportStatisticsSnapshot *baseline =
      impl_->emittedStatistics ? &*impl_->emittedStatistics : nullptr;
  const std::size_t deploymentOffset =
      baseline ? baseline->deployment.size() : 0;
  if (deploymentOffset > current.deployment.size())
    llvm_unreachable("execution-matrix deployment statistics regressed");
  emitDeploymentOperationRows(
      invocation,
      llvm::ArrayRef(current.deployment).drop_front(deploymentOffset));
  emitCacheRow(invocation, cacheRow(current.artifact,
                                    baseline ? &baseline->artifact : nullptr));
  emitCacheRow(invocation, cacheRow(current.fabric,
                                    baseline ? &baseline->fabric : nullptr));
  emitCacheRow(invocation,
               cacheRow(current.configurationAbi,
                        baseline ? &baseline->configurationAbi : nullptr));
  emitCacheRow(invocation,
               cacheRow(current.systemMapping,
                        baseline ? &baseline->systemMapping : nullptr));
  emitCacheRow(
      invocation,
      cacheRow(current.configurationProjection,
               baseline ? &baseline->configurationProjection : nullptr));
  emitCacheRow(invocation, cacheRow(current.gem5Facts,
                                    baseline ? &baseline->gem5Facts : nullptr));
  emitCacheRow(invocation,
               externalFileFingerprintRow(
                   current.gem5Facts, baseline ? &baseline->gem5Facts : nullptr));
  const runtime::Gem5SystemFactsConstructionStatistics *priorConstruction =
      baseline ? &baseline->gem5Facts.construction : nullptr;
  emitFactsOperationRow(
      invocation, "derive_facts",
      operationDelta(current.gem5Facts.construction.deriveFacts,
                     priorConstruction ? &priorConstruction->deriveFacts
                                       : nullptr));
  emitFactsOperationRow(
      invocation, "system_inputs_and_deployment_import",
      operationDelta(
          current.gem5Facts.construction.systemInputsAndDeploymentImport,
          priorConstruction
              ? &priorConstruction->systemInputsAndDeploymentImport
              : nullptr));
  emitFactsOperationRow(
      invocation, "gem5_binding_import",
      operationDelta(current.gem5Facts.construction.bindingImport,
                     priorConstruction ? &priorConstruction->bindingImport
                                       : nullptr));
  emitFactsOperationRow(
      invocation, "entire_fabric_root_import",
      operationDelta(current.gem5Facts.construction.fabricImport,
                     priorConstruction ? &priorConstruction->fabricImport
                                       : nullptr));
  emitFactsOperationRow(
      invocation, "system_mapping_import",
      operationDelta(current.gem5Facts.construction.systemMappingImport,
                     priorConstruction ? &priorConstruction->systemMappingImport
                                       : nullptr));
  emitFactsOperationRow(
      invocation, "gem5_guest_runtime_image_projection",
      operationDelta(current.gem5Facts.construction.guestRuntimeImageProjection,
                     priorConstruction
                         ? &priorConstruction->guestRuntimeImageProjection
                         : nullptr));
  impl_->emittedStatistics = std::move(current);
}

} // namespace loom::system_test
