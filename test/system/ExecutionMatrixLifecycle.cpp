#include "ExecutionMatrixLifecycle.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Deployment/DeploymentDiagnostics.h"
#include "Deployment/HardwareConfigurationImage.h"
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
inline constexpr std::size_t systemRtlBuildCommandCount = 4;

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

struct ResourceSnapshot final {
  std::chrono::steady_clock::time_point wall;
  std::optional<std::uint64_t> selfCpuNanoseconds;
  std::optional<std::uint64_t> childCpuNanoseconds;
  std::optional<std::uint64_t> selfProcessLifetimeHighWaterRssKib;
  std::optional<std::uint64_t> maximumWaitedDescendantProcessRssKib;
};

ResourceSnapshot captureResources() {
  ResourceSnapshot snapshot;
  snapshot.wall = std::chrono::steady_clock::now();
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
  return commandOrdinal < systemRtlBuildCommandCount ? "rtl_compile"
                                                     : "rtl_controller";
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
               << " schema=loom.execution_matrix_summary.3";
  emitInvocationKey(llvm::outs(), invocation);
  llvm::outs() << " deterministic_work=" << deterministicWork
               << " maximum_waited_descendant_process_rss_kib="
               << maximumWaitedDescendantProcessRssKib;
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
  };

  std::uint64_t reserveSequence() { return nextSequence_++; }

  void record(std::uint64_t sequence,
              ExecutionMatrixLifecycleOperation operation,
              const ResourceSnapshot &begin, const ResourceSnapshot &end) {
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end.wall - begin.wall);
    records_.push_back(
        {sequence, operation,
         static_cast<std::uint64_t>(std::max<std::int64_t>(0, wall.count())),
         difference(end.selfCpuNanoseconds, begin.selfCpuNanoseconds),
         difference(end.childCpuNanoseconds, begin.childCpuNanoseconds),
         end.selfProcessLifetimeHighWaterRssKib,
         end.maximumWaitedDescendantProcessRssKib});
  }

  void emit(const ExecutionMatrixInvocation &invocation) const {
    std::vector<Record> ordered = records_;
    llvm::sort(ordered, [](const Record &lhs, const Record &rhs) {
      return lhs.sequence < rhs.sequence;
    });
    for (const Record &record : ordered) {
      llvm::outs() << "execution-matrix-lifecycle"
                   << " schema=loom.execution_matrix_lifecycle.4.0";
      emitInvocationKey(llvm::outs(), invocation);
      llvm::outs() << " interval_kind=inclusive parent="
                   << lifecycleParent(record.operation)
                   << " operation=" << spelling(record.operation)
                   << " wall_ns=" << record.wallNanoseconds << " self_cpu_ns=";
      printOptional(llvm::outs(), record.selfCpuNanoseconds);
      llvm::outs() << " child_cpu_ns=";
      printOptional(llvm::outs(), record.childCpuNanoseconds);
      llvm::outs() << " self_process_lifetime_high_water_rss_kib_snapshot=";
      printOptional(llvm::outs(), record.selfProcessLifetimeHighWaterRssKib);
      llvm::outs() << " maximum_waited_descendant_process_rss_kib_snapshot=";
      printOptional(llvm::outs(), record.maximumWaitedDescendantProcessRssKib);
      llvm::outs() << '\n';
    }
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

void emitExecutionMatrixExternalCommands(
    ExecutionMatrixInvocation invocation,
    llvm::ArrayRef<external_tool::ExternalToolCommandExecutionObservation>
        commands) {
  if (invocation.cell != ExecutionMatrixCell::SystemRtl)
    return;
  constexpr std::size_t expectedCommandCount = systemRtlBuildCommandCount + 1;
  if (commands.size() != expectedCommandCount)
    llvm::report_fatal_error(
        "System RTL execution did not report four builds and one controller");
  for (const auto indexed : llvm::enumerate(commands)) {
    const external_tool::ExternalToolCommandExecutionObservation &command =
        indexed.value();
    if (command.commandOrdinal != indexed.index() || command.exitCode != 0)
      llvm::report_fatal_error(
          "System RTL command observations are not canonical successes");
    llvm::outs() << "execution-matrix-external-command"
                 << " schema=loom.execution_matrix_external_command.2";
    emitInvocationKey(llvm::outs(), invocation);
    llvm::outs() << " command_ordinal=" << command.commandOrdinal
                 << " command_role=" << systemRtlCommandRole(indexed.index())
                 << " wall_ns=" << command.wallNanoseconds
                 << " exit_code=" << command.exitCode
                 << " total_build_jobs=" << LOOM_TEST_BUILD_JOBS
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
  const ResourceSnapshot end = captureResources();
  impl_->recorder->impl_->record(impl_->sequence, impl_->operation,
                                 impl_->begin, end);
}

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

  deployment::DeploymentConstructionStatisticsSession deploymentConstruction;
  evaluation::ArtifactImportCacheScope artifactImports;
  fabric::FabricArtifactImportSession fabricImports;
  hardware::ConfigurationABIImportSession configurationAbiImports;
  mapping::SystemMappingImportSession systemMappingImports;
  deployment::ConfigurationImageProjectionSession configurationProjections;
  runtime::Gem5SystemFactsSession gem5Facts;
};

ExecutionMatrixImportSessions::ExecutionMatrixImportSessions(
    const ArtifactStore &artifacts, const BlobStore &blobs)
    : impl_(std::make_unique<Impl>(artifacts, blobs)) {}
ExecutionMatrixImportSessions::~ExecutionMatrixImportSessions() = default;

ExecutionMatrixImportSummary ExecutionMatrixImportSessions::summary() const {
  const evaluation::ArtifactImportCacheStatistics artifact =
      impl_->artifactImports.statistics();
  const fabric::FabricArtifactImportSessionStatistics fabric =
      impl_->fabricImports.statistics();
  const hardware::ConfigurationABIImportSessionStatistics abi =
      impl_->configurationAbiImports.statistics();
  const mapping::SystemMappingImportSessionStatistics mapping =
      impl_->systemMappingImports.statistics();
  const deployment::ConfigurationImageProjectionSessionStatistics projection =
      impl_->configurationProjections.statistics();
  const runtime::Gem5SystemFactsSessionStatistics facts =
      impl_->gem5Facts.statistics();
  return {facts.requests,
          facts.cacheHits,
          facts.cacheMisses,
          facts.uniqueConstructions,
          facts.revalidatedArtifactBytes,
          facts.revalidatedBlobBytes,
          facts.constructionNanosecondsSaved,
          artifact.cacheHits,
          fabric.cacheHits,
          abi.cacheHits,
          mapping.cacheHits,
          projection.cacheHits};
}

bool ExecutionMatrixImportSessions::reusedOneExactGem5FactsClosure() const {
  const ExecutionMatrixImportSummary imports = summary();
  return imports.gem5FactsRequests == 2 && imports.gem5FactsMisses == 1 &&
         imports.gem5FactsUniqueConstructions == 1 &&
         imports.gem5FactsHits == 1 &&
         imports.gem5FactsHits + imports.gem5FactsMisses ==
             imports.gem5FactsRequests &&
         imports.gem5FactsRevalidatedArtifactBytes != 0 &&
         imports.gem5FactsRevalidatedBlobBytes != 0 &&
         imports.gem5FactsConstructionNanosecondsSaved != 0 &&
         imports.artifactImportHits != 0 && imports.fabricImportHits != 0 &&
         imports.configurationAbiImportHits != 0 &&
         imports.systemMappingImportHits != 0 &&
         imports.configurationProjectionHits != 0;
}

void ExecutionMatrixImportSessions::emitStatistics(
    const ExecutionMatrixInvocation &invocation) const {
  emitDeploymentOperationRows(invocation,
                              impl_->deploymentConstruction.statistics());
  const evaluation::ArtifactImportCacheStatistics artifact =
      impl_->artifactImports.statistics();
  emitCacheRow(invocation,
               {"artifact_import", "artifact_revalidation",
                artifact.importRequests, artifact.cacheHits,
                artifact.cacheMisses, artifact.cacheMisses,
                artifact.uniqueConstructions, artifact.uncachedConstructions, 0,
                0, artifact.revalidationCount, artifact.revalidatedBytes, 0,
                artifact.constructionNanoseconds, artifact.minimumRetainedBytes,
                artifact.entryCount});

  const fabric::FabricArtifactImportSessionStatistics fabric =
      impl_->fabricImports.statistics();
  emitCacheRow(invocation,
               {"fabric_import", "artifact_revalidation", fabric.importRequests,
                fabric.cacheHits, fabric.cacheMisses, fabric.cacheMisses,
                fabric.uniqueConstructions, fabric.uncachedConstructions, 0, 0,
                fabric.revalidationCount, fabric.revalidatedBytes, 0,
                fabric.constructionNanoseconds, fabric.retainedPayloadBytes,
                fabric.entryCount});

  const hardware::ConfigurationABIImportSessionStatistics abi =
      impl_->configurationAbiImports.statistics();
  emitCacheRow(invocation,
               {"configuration_abi_import", "immutable_session_domain",
                abi.importRequests, abi.cacheHits, abi.cacheMisses,
                abi.cacheMisses, abi.uniqueConstructions, 0, 0, 0, 0, 0, 0,
                abi.constructionNanoseconds, abi.retainedBytes,
                abi.entryCount});

  const mapping::SystemMappingImportSessionStatistics mapping =
      impl_->systemMappingImports.statistics();
  emitCacheRow(invocation,
               {"system_mapping_import", "immutable_session_domain",
                mapping.importRequests, mapping.cacheHits, mapping.cacheMisses,
                mapping.cacheMisses, mapping.uniqueConstructions,
                mapping.uncachedConstructions, 0, 0, 0, 0, 0,
                mapping.constructionNanoseconds, mapping.retainedBytes,
                mapping.entryCount});

  const deployment::ConfigurationImageProjectionSessionStatistics projection =
      impl_->configurationProjections.statistics();
  emitCacheRow(invocation,
               {"configuration_image_projection", "immutable_session_domain",
                projection.requests, projection.cacheHits,
                projection.cacheMisses, projection.cacheMisses,
                projection.uniqueConstructions,
                projection.uncachedConstructions, 0, 0, 0, 0, 0,
                projection.constructionNanoseconds, projection.retainedBytes,
                projection.entryCount});

  const runtime::Gem5SystemFactsSessionStatistics facts =
      impl_->gem5Facts.statistics();
  emitCacheRow(invocation,
               {"gem5_system_facts", "complete_closure_revalidation",
                facts.requests, facts.cacheHits, facts.cacheMisses,
                facts.constructionAttempts, facts.uniqueConstructions,
                facts.uncachedConstructions, facts.unsupportedConstructions,
                facts.failedConstructions, facts.revalidationCount,
                facts.revalidatedArtifactBytes, facts.revalidatedBlobBytes,
                facts.constructionNanoseconds, facts.minimumRetainedBytes,
                facts.entryCount});
  emitFactsOperationRow(invocation, "derive_facts",
                        facts.construction.deriveFacts);
  emitFactsOperationRow(invocation, "system_inputs_and_deployment_import",
                        facts.construction.systemInputsAndDeploymentImport);
  emitFactsOperationRow(invocation, "gem5_binding_import",
                        facts.construction.bindingImport);
  emitFactsOperationRow(invocation, "entire_fabric_root_import",
                        facts.construction.fabricImport);
  emitFactsOperationRow(invocation, "system_mapping_import",
                        facts.construction.systemMappingImport);
  emitFactsOperationRow(invocation, "gem5_guest_runtime_image_projection",
                        facts.construction.guestRuntimeImageProjection);
}

} // namespace loom::system_test
