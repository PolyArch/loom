#ifndef LOOM_APPLICATION_BUILDDIAGNOSTICS_H
#define LOOM_APPLICATION_BUILDDIAGNOSTICS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>

namespace loom::dse {
struct IncompletePreMappingExploration;
struct ResourceTimeMappingFunnel;
} // namespace loom::dse

namespace loom::application {

struct PreparedApplicationBuild;
struct ApplicationMappingExecution;
struct ApplicationPairDecisionRecord;

enum class ApplicationBuildOperation : std::uint8_t {
  ProductTargetPreparation,
  FinalLinkImport,
  ApplicationPreparation,
  MappingExecution,
  MappingImport,
  ConfigurationAbiDerivation,
  HardwareBindingDerivation,
  CompilerTargetResolution,
  HostProgramFinalization,
  InstructionBinaryFinalization,
  DeclarativeDeploymentFinalization,
  DeploymentConstruction,
  PackagePublication,
};

struct ApplicationBuildOperationStatistics final {
  ApplicationBuildOperation operation;
  std::uint64_t durationNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
};

struct ApplicationMappingExecutionPolicyStatistics final {
  std::uint64_t requestedWallTimeLimitMilliseconds = 0;
  std::uint64_t dispatchNotAfterUnixNanoseconds = 0;
  std::uint64_t observedWallTimeNanoseconds = 0;
  bool deadlineObserved = false;
};

void emitApplicationBuildOperationStatistics(
    const ApplicationBuildOperationStatistics &statistics);

void emitApplicationMappingExecutionPolicyStatistics(
    const ApplicationMappingExecutionPolicyStatistics &statistics);

/// Emits the bounded pre-Mapping policy, deterministic work accounting, and
/// candidate inventory through the process-wide diagnostic stream. The typed
/// build objects remain the semantic owner; JSON exists only at this
/// presentation boundary.
void emitApplicationPlanningDiagnostics(
    const PreparedApplicationBuild &prepared);

/// Emits a terminal resource-time funnel checkpoint through the same JSON
/// projection used by the completed application-planning record.
void emitApplicationResourceTimeFunnelTerminalDiagnostics(
    const dse::ResourceTimeMappingFunnel &funnel, llvm::StringRef status);

void emitApplicationPreMappingIncompleteDiagnostics(
    const dse::IncompletePreMappingExploration &incomplete);

/// Emits the exact planning-record to Mapping-outcome join retained by one
/// application build. Incomplete, infeasible, and verified outcomes remain
/// distinct in the diagnostic representation.
void emitApplicationMappingDiagnostics(
    const ApplicationMappingExecution &execution);

/// Emits a pair-level decision when preparation terminates before a complete
/// ApplicationMappingExecution exists. The detailed planning/checkpoint
/// diagnostics remain emitted by their existing owners.
void emitApplicationPairDecisionDiagnostics(
    const ApplicationPairDecisionRecord &decision);

/// Shared presentation-only projection used by diagnostics and product
/// visualization. The decision record remains the semantic owner.
llvm::json::Object projectApplicationPairDecisionJson(
    const ApplicationPairDecisionRecord &decision);

} // namespace loom::application

#endif // LOOM_APPLICATION_BUILDDIAGNOSTICS_H
