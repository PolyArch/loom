#ifndef LOOM_COMMON_INVOCATIONDIAGNOSTICLOG_H
#define LOOM_COMMON_INVOCATIONDIAGNOSTICLOG_H

#include "Common/DiagnosticVerbosity.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/JSON.h"

#include <cstdint>

namespace loom {

/// Closed stages represented by the process-wide invocation diagnostic
/// stream. Domain serializers own payload meaning; Common owns the envelope.
enum class InvocationDiagnosticStage : std::uint8_t {
  DataflowLowering,
  TechMapping,
  SpatialPnr,
  SystemPnr,
  HardwareConfiguration,
  Deployment,
};

/// Closed events represented by the invocation diagnostic schema. Payload
/// field names and types are part of the schema and remain owned by the event's
/// domain serializer.
enum class InvocationDiagnosticEvent : std::uint8_t {
  InvocationBegin,
  InvocationEnd,
  Statistics,
  Candidate,
  Seed,
  NegotiationIteration,
  CapacityConflict,
  ActionProposal,
  ActionOutcome,
  ContextChoice,
  NetRoute,
  CutAnalysis,
  DerivedContext,
  TopologyQuality,
  TagDomainPressure,
  ArithmeticFailure,
  MappingFailure,
  ConfigurationAbiDerivation,
  ConfigurationAbiConstruction,
  ConfigurationAbiImportSession,
  ConfigurationImageProjectionSession,
  SystemMappingImportSession,
  ArtifactImportSession,
  Gem5SystemFactsSession,
  ApplicationBuildStatistics,
  DeploymentConstructionStatistics,
  DeploymentPackageStatistics,
};

bool invocationDiagnosticEnabled(DiagnosticVerbosity minimum);

/// Emits one line-atomic JSONL record. The schema covers both the Common-owned
/// envelope and each registered event's owner-serialized payload. The builder
/// is not invoked when the requested verbosity is disabled.
void emitInvocationDiagnostic(
    DiagnosticVerbosity minimum, InvocationDiagnosticStage stage,
    InvocationDiagnosticEvent event,
    llvm::function_ref<llvm::json::Value()> buildPayload);

} // namespace loom

#endif // LOOM_COMMON_INVOCATIONDIAGNOSTICLOG_H
