#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXLIFECYCLE_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXLIFECYCLE_H

#include "ExecutionMatrixInvocation.h"
#include "ExternalTool/InvocationBundle.h"
#include "Runtime/Gem5SystemExecution.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <variant>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::eda::test {
enum class MappedSpatialHardwareFixtureOperation : std::uint8_t;
} // namespace loom::eda::test

namespace loom::system_test {

llvm::Error emitExecutionMatrixRunSummary(
    const ExecutionMatrixInvocation &invocation,
    std::uint64_t deterministicWork,
    std::uint64_t maximumWaitedDescendantProcessRssKib,
    const runtime::Gem5SystemAttemptProfile *profile,
    llvm::ArrayRef<runtime::Gem5SpatialInvocationProjection>
        spatialInvocations);

enum class ExecutionMatrixLifecycleOperation : std::uint8_t {
  Setup,
  DataflowConstructionAndPublication,
  FabricModuleConstructionAndFinalization,
  TechMapping,
  SpatialPnr,
  SystemFabricAndInterconnectConstruction,
  ConfigurationAbiAndHardwareImplementationGeneration,
  SystemMappingAndPnr,
  GuestCompileAndLink,
  RuntimeBindingAndDeploymentFinalization,
  WorkloadAndRuntimeInputPublication,
  HostLifecycle,
  Gem5Readiness,
  Gem5Binding,
  RequestConstruction,
  OrdinaryPrepare,
  OrdinaryExternalExecution,
  OrdinaryImportAndEvidenceAssembly,
  OrdinaryEvidencePublication,
  OrdinaryExecutionImport,
  DiagnosticPrepare,
  DiagnosticExternalExecution,
  DiagnosticImportAndEvidenceAssembly,
  DiagnosticEvidencePublication,
  DiagnosticExecutionImport,
  OracleVerification,
  Cleanup,
};

class ExecutionMatrixLifecycleRecorder final {
public:
  class Impl;

  ExecutionMatrixLifecycleRecorder();
  ~ExecutionMatrixLifecycleRecorder();

  ExecutionMatrixLifecycleRecorder(const ExecutionMatrixLifecycleRecorder &) =
      delete;
  ExecutionMatrixLifecycleRecorder &
  operator=(const ExecutionMatrixLifecycleRecorder &) = delete;

  void emit(const ExecutionMatrixInvocation &invocation) const;
  void emitAttemptPair(ExecutionMatrixCell cell) const;
  std::uint64_t
  operationCount(ExecutionMatrixLifecycleOperation operation) const;

private:
  std::unique_ptr<Impl> impl_;
  friend class ExecutionMatrixLifecycleTimer;
};

class ExecutionMatrixLifecycleTimer final {
public:
  class Impl;

  ExecutionMatrixLifecycleTimer(ExecutionMatrixLifecycleRecorder &recorder,
                                ExecutionMatrixLifecycleOperation operation);
  ExecutionMatrixLifecycleTimer(
      ExecutionMatrixLifecycleRecorder &recorder,
      eda::test::MappedSpatialHardwareFixtureOperation operation);
  ~ExecutionMatrixLifecycleTimer();

  std::uint64_t finish();

  ExecutionMatrixLifecycleTimer(const ExecutionMatrixLifecycleTimer &) = delete;
  ExecutionMatrixLifecycleTimer &
  operator=(const ExecutionMatrixLifecycleTimer &) = delete;

private:
  std::unique_ptr<Impl> impl_;
};

struct ExecutionMatrixImportSummary final {
  std::uint64_t gem5FactsRequests = 0;
  std::uint64_t gem5FactsHits = 0;
  std::uint64_t gem5FactsMisses = 0;
  std::uint64_t gem5FactsConstructionAttempts = 0;
  std::uint64_t gem5FactsUniqueConstructions = 0;
  std::uint64_t gem5FactsUncachedConstructions = 0;
  std::uint64_t gem5FactsUnsupportedConstructions = 0;
  std::uint64_t gem5FactsFailedConstructions = 0;
  std::uint64_t gem5FactsRevalidationCount = 0;
  std::uint64_t gem5FactsRevalidatedArtifactBytes = 0;
  std::uint64_t gem5FactsRevalidatedBlobBytes = 0;
  std::uint64_t gem5FactsConstructionNanosecondsSaved = 0;
  std::uint64_t gem5FactsEntryCount = 0;
  std::uint64_t artifactImportHits = 0;
  std::uint64_t fabricImportHits = 0;
  std::uint64_t configurationAbiImportHits = 0;
  std::uint64_t systemMappingImportHits = 0;
  std::uint64_t configurationProjectionHits = 0;
};

/// The make job budget one full-budget RTL cell hands to the Verilator
/// provider: the largest admitted mapped-RTL parallelism count within the
/// configured host job budget.
std::uint64_t fullBudgetRtlBuildJobs();

/// The System RTL command schedule: one Verilation and one archive build per
/// spatial launch, then the gem5 controller.
inline constexpr std::size_t systemRtlSpatialLaunchCount = 4;
inline constexpr std::size_t systemRtlCommandCount =
    2 * systemRtlSpatialLaunchCount + 1;

void emitExecutionMatrixExternalCommands(
    ExecutionMatrixInvocation invocation,
    llvm::ArrayRef<external_tool::ExternalToolCommandExecutionObservation>
        commands);

/// The shared setup of one attempt pair: the fixture, the gem5 readiness
/// record, and the gem5 binding constructed once for both attempts.
struct ExecutionMatrixAttemptPairScope final {
  ExecutionMatrixCell cell;
};

/// The observer of one import-statistics delta: one attempt, or the shared
/// setup of one attempt pair.
using ExecutionMatrixRowScope =
    std::variant<ExecutionMatrixInvocation, ExecutionMatrixAttemptPairScope>;

/// Owns every removable immutable import/projection session for one exact
/// execution-matrix store domain. The sessions live for the store's lifetime,
/// so the fixture construction and every attempt construct each closure once
/// and reuse it after revalidation. Independent replay cells construct
/// isolated instances and never share session attachments.
class ExecutionMatrixImportSessions final {
public:
  class Impl;

  ExecutionMatrixImportSessions(const ArtifactStore &artifacts,
                                const BlobStore &blobs);
  ~ExecutionMatrixImportSessions();

  ExecutionMatrixImportSessions(const ExecutionMatrixImportSessions &) = delete;
  ExecutionMatrixImportSessions &
  operator=(const ExecutionMatrixImportSessions &) = delete;

  ExecutionMatrixImportSummary summary() const;
  bool reusedOneExactGem5FactsClosure() const;
  bool reusedOneExactGem5FactsClosureAcrossAttemptPair() const;
  /// Emits the counter deltas observed since the previous emission under
  /// the given scope's key.
  void emitStatistics(const ExecutionMatrixRowScope &scope);

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXLIFECYCLE_H
