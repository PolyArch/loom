#ifndef LOOM_COMMON_MAPPINGDEBUGLOG_H
#define LOOM_COMMON_MAPPINGDEBUGLOG_H

#include "Common/DiagnosticVerbosity.h"
#include "Common/InvocationDiagnosticLog.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

namespace loom::mapping_debug {

using Level = DiagnosticVerbosity;
using Stage = InvocationDiagnosticStage;
using Event = InvocationDiagnosticEvent;

enum class ClosureStatus : std::uint8_t {
  Internal,
  SearchExhausted,
  SemanticLimitReached,
  CancelledOrTimeout,
  ProvenInfeasible,
  ProofNotEstablished,
  Invalid,
  Failed,
  ArithmeticFailure,
  RouteFailure,
  RouteTerminalMismatch,
  MappingNonclosure,
  SelectedHandshakeCycle,
  Closed,
  TemporaryMapping,
  FixedTerminalCutTemporary,
  FixedTerminalCut,
  NoProgressTemporary,
  NoProgress,
  TemporaryCapacity,
  IterationLimit,
};

llvm::StringRef closureStatusSpelling(ClosureStatus status);

/// Returns the Common-owned process-wide diagnostic level.
Level level();

bool enabled(Level minimum);

/// Emits one line-atomic JSON event. The builder owns only the nested payload;
/// Common owns every envelope field. The builder is not invoked when the
/// requested level is disabled.
void emit(Level minimum, Stage stage, Event event,
          llvm::function_ref<void(llvm::json::Object &)> buildFields = {});

struct MappingRunStatistics final {
  std::uint64_t candidateRows = 0;
  std::uint64_t candidatePublications = 0;
  std::uint64_t actionsProposed = 0;
  std::uint64_t actionsAccepted = 0;
  std::uint64_t actionsRejected = 0;
  std::uint64_t actionsRolledBack = 0;
  std::uint64_t aStarExpansions = 0;
  std::uint64_t negotiatedIterations = 0;
  std::uint64_t capacityConflicts = 0;
  std::uint64_t arithmeticFailures = 0;

  void emit(Stage stage, ClosureStatus closureStatus) const;
  void emit(Stage stage, ClosureStatus closureStatus,
            llvm::function_ref<void(llvm::json::Object &)> buildFields) const;
};

} // namespace loom::mapping_debug

#endif // LOOM_COMMON_MAPPINGDEBUGLOG_H
