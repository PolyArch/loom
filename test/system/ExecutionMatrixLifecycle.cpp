#include "ExecutionMatrixLifecycle.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Runtime/Gem5SystemExecution.h"

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
  if (seconds >
      (std::numeric_limits<std::uint64_t>::max() - subsecond) /
          nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + subsecond;
}

struct ResourceSnapshot final {
  std::chrono::steady_clock::time_point wall;
  std::optional<std::uint64_t> selfCpuNanoseconds;
  std::optional<std::uint64_t> childCpuNanoseconds;
  std::optional<std::uint64_t> selfPeakRssKib;
  std::optional<std::uint64_t> childPeakRssKib;
};

ResourceSnapshot captureResources() {
  ResourceSnapshot snapshot;
  snapshot.wall = std::chrono::steady_clock::now();
  snapshot.selfCpuNanoseconds = processCpuNanoseconds();
  rusage selfUsage{};
  if (::getrusage(RUSAGE_SELF, &selfUsage) == 0 && selfUsage.ru_maxrss >= 0)
    snapshot.selfPeakRssKib = selfUsage.ru_maxrss;
  rusage usage{};
  if (::getrusage(RUSAGE_CHILDREN, &usage) == 0) {
    auto user = timevalNanoseconds(usage.ru_utime);
    auto system = timevalNanoseconds(usage.ru_stime);
    if (user && system &&
        *system <= std::numeric_limits<std::uint64_t>::max() - *user)
      snapshot.childCpuNanoseconds = *user + *system;
    if (usage.ru_maxrss >= 0)
      snapshot.childPeakRssKib = usage.ru_maxrss;
  }
  return snapshot;
}

std::optional<std::uint64_t>
difference(std::optional<std::uint64_t> end,
           std::optional<std::uint64_t> begin) {
  if (!end || !begin || *end < *begin)
    return std::nullopt;
  return *end - *begin;
}

llvm::StringRef spelling(ExecutionMatrixLifecycleOperation operation) {
  switch (operation) {
  case ExecutionMatrixLifecycleOperation::Setup:
    return "setup";
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

void emitCacheRow(llvm::StringRef cell, const CacheRow &row) {
  llvm::outs() << "execution-matrix-cache"
               << " schema=loom.execution_matrix_cache.1"
               << " cell=" << cell << " cache=" << row.cache
               << " hit_validation=" << row.hitValidation
               << " requests=" << row.requests << " hits=" << row.hits
               << " misses=" << row.misses
               << " construction_attempts=" << row.constructionAttempts
               << " unique_constructions=" << row.uniqueConstructions
               << " uncached_constructions=" << row.uncachedConstructions
               << " unsupported_constructions="
               << row.unsupportedConstructions
               << " failed_constructions=" << row.failedConstructions
               << " revalidation_count=" << row.revalidationCount
               << " revalidated_artifact_bytes="
               << row.revalidatedArtifactBytes
               << " revalidated_blob_bytes=" << row.revalidatedBlobBytes
               << " construction_wall_ns=" << row.constructionNanoseconds
               << " minimum_retained_bytes=" << row.minimumRetainedBytes
               << " entries=" << row.entryCount << '\n';
}

} // namespace

class ExecutionMatrixLifecycleRecorder::Impl final {
public:
  struct Record final {
    std::uint64_t sequence = 0;
    ExecutionMatrixLifecycleOperation operation;
    std::uint64_t wallNanoseconds = 0;
    std::optional<std::uint64_t> selfCpuNanoseconds;
    std::optional<std::uint64_t> childCpuNanoseconds;
    std::optional<std::uint64_t> selfPeakRssKib;
    std::optional<std::uint64_t> childPeakRssKib;
  };

  std::uint64_t reserveSequence() { return nextSequence_++; }

  void record(std::uint64_t sequence,
              ExecutionMatrixLifecycleOperation operation,
              const ResourceSnapshot &begin, const ResourceSnapshot &end) {
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end.wall - begin.wall);
    records_.push_back(
        {sequence,
         operation,
         static_cast<std::uint64_t>(std::max<std::int64_t>(0, wall.count())),
         difference(end.selfCpuNanoseconds, begin.selfCpuNanoseconds),
         difference(end.childCpuNanoseconds, begin.childCpuNanoseconds),
         end.selfPeakRssKib,
         end.childPeakRssKib});
  }

  void emit(llvm::StringRef cell) const {
    std::vector<Record> ordered = records_;
    llvm::sort(ordered, [](const Record &lhs, const Record &rhs) {
      return lhs.sequence < rhs.sequence;
    });
    for (const Record &record : ordered) {
      llvm::outs() << "execution-matrix-lifecycle"
                   << " schema=loom.execution_matrix_lifecycle.1"
                   << " cell=" << cell
                   << " operation=" << spelling(record.operation)
                   << " wall_ns=" << record.wallNanoseconds
                   << " self_cpu_ns=";
      printOptional(llvm::outs(), record.selfCpuNanoseconds);
      llvm::outs() << " child_cpu_ns=";
      printOptional(llvm::outs(), record.childCpuNanoseconds);
      llvm::outs() << " self_process_lifetime_peak_rss_kib=";
      printOptional(llvm::outs(), record.selfPeakRssKib);
      llvm::outs() << " child_process_lifetime_peak_rss_kib=";
      printOptional(llvm::outs(), record.childPeakRssKib);
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
ExecutionMatrixLifecycleRecorder::~ExecutionMatrixLifecycleRecorder() =
    default;

void ExecutionMatrixLifecycleRecorder::emit(llvm::StringRef cell) const {
  impl_->emit(cell);
}

ExecutionMatrixLifecycleTimer::ExecutionMatrixLifecycleTimer(
    ExecutionMatrixLifecycleRecorder &recorder,
    ExecutionMatrixLifecycleOperation operation)
    : impl_(std::make_unique<Impl>(recorder, operation,
                                  recorder.impl_->reserveSequence())) {}

ExecutionMatrixLifecycleTimer::~ExecutionMatrixLifecycleTimer() {
  const ResourceSnapshot end = captureResources();
  impl_->recorder->impl_->record(impl_->sequence, impl_->operation, impl_->begin,
                                end);
}

class ExecutionMatrixImportSessions::Impl final {
public:
  Impl(const ArtifactStore &artifacts, const BlobStore &blobs)
      : artifactImports(artifacts, &blobs, artifactImportEntryLimit),
        fabricImports(fabric::FabricArtifactImportSessionMode::Isolated,
                      fabricImportEntryLimit),
        configurationAbiImports(
            hardware::ConfigurationABIImportSessionMode::Isolated),
        systemMappingImports(artifacts, systemMappingImportEntryLimit),
        configurationProjections(artifacts,
                                 configurationProjectionEntryLimit),
        gem5Facts(artifacts, blobs,
                  runtime::Gem5SystemFactsSessionMode::Isolated,
                  gem5FactsEntryLimit) {}

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
  const runtime::Gem5SystemFactsSessionStatistics facts =
      impl_->gem5Facts.statistics();
  return {facts.requests, facts.cacheHits, facts.cacheMisses,
          facts.uniqueConstructions};
}

void ExecutionMatrixImportSessions::emitStatistics(llvm::StringRef cell) const {
  const evaluation::ArtifactImportCacheStatistics artifact =
      impl_->artifactImports.statistics();
  emitCacheRow(cell,
               {"artifact_import", "artifact_revalidation",
                artifact.importRequests, artifact.cacheHits,
                artifact.cacheMisses, artifact.cacheMisses,
                artifact.uniqueConstructions, artifact.uncachedConstructions,
                0, 0, artifact.revalidationCount,
                artifact.revalidatedBytes, 0,
                artifact.constructionNanoseconds,
                artifact.minimumRetainedBytes, artifact.entryCount});

  const fabric::FabricArtifactImportSessionStatistics fabric =
      impl_->fabricImports.statistics();
  emitCacheRow(cell,
                {"fabric_import", "artifact_revalidation",
                fabric.importRequests, fabric.cacheHits, fabric.cacheMisses,
                fabric.cacheMisses, fabric.uniqueConstructions,
                fabric.uncachedConstructions, 0, 0, fabric.revalidationCount,
                fabric.revalidatedBytes, 0,
                fabric.constructionNanoseconds, fabric.retainedPayloadBytes,
                fabric.entryCount});

  const hardware::ConfigurationABIImportSessionStatistics abi =
      impl_->configurationAbiImports.statistics();
  emitCacheRow(cell,
                {"configuration_abi_import", "immutable_session_domain",
                abi.importRequests, abi.cacheHits, abi.cacheMisses,
                abi.cacheMisses, abi.uniqueConstructions, 0, 0, 0, 0, 0, 0,
                abi.constructionNanoseconds, abi.retainedBytes,
                abi.entryCount});

  const mapping::SystemMappingImportSessionStatistics mapping =
      impl_->systemMappingImports.statistics();
  emitCacheRow(cell,
               {"system_mapping_import", "immutable_session_domain",
                mapping.importRequests, mapping.cacheHits,
                mapping.cacheMisses, mapping.cacheMisses,
                mapping.uniqueConstructions, mapping.uncachedConstructions, 0,
                0, 0, 0, 0,
                mapping.constructionNanoseconds, mapping.retainedBytes,
                mapping.entryCount});

  const deployment::ConfigurationImageProjectionSessionStatistics projection =
      impl_->configurationProjections.statistics();
  emitCacheRow(cell,
               {"configuration_image_projection", "immutable_session_domain",
                projection.requests, projection.cacheHits,
                projection.cacheMisses, projection.cacheMisses,
                projection.uniqueConstructions,
                projection.uncachedConstructions, 0, 0, 0, 0, 0,
                projection.constructionNanoseconds, projection.retainedBytes,
                projection.entryCount});

  const runtime::Gem5SystemFactsSessionStatistics facts =
      impl_->gem5Facts.statistics();
  emitCacheRow(cell,
                {"gem5_system_facts", "complete_closure_revalidation",
                facts.requests, facts.cacheHits, facts.cacheMisses,
                facts.constructionAttempts, facts.uniqueConstructions,
                facts.uncachedConstructions, facts.unsupportedConstructions,
                facts.failedConstructions, facts.revalidationCount,
                facts.revalidatedArtifactBytes,
                facts.revalidatedBlobBytes, facts.constructionNanoseconds,
                facts.minimumRetainedBytes, facts.entryCount});
}

} // namespace loom::system_test
