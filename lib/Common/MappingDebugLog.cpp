#include "Common/MappingDebugLog.h"

#include "Common/InvocationDiagnosticLog.h"

namespace loom::mapping_debug {
namespace {

} // namespace

Level level() { return diagnosticVerbosity(); }

bool enabled(Level minimum) { return invocationDiagnosticEnabled(minimum); }

llvm::StringRef closureStatusSpelling(ClosureStatus status) {
  switch (status) {
  case ClosureStatus::Internal:
    return "internal";
  case ClosureStatus::SearchExhausted:
    return "search_exhausted";
  case ClosureStatus::SemanticLimitReached:
    return "semantic_limit_reached";
  case ClosureStatus::CancelledOrTimeout:
    return "cancelled_or_timeout";
  case ClosureStatus::ProvenInfeasible:
    return "proven_infeasible";
  case ClosureStatus::ProofNotEstablished:
    return "proof_not_established";
  case ClosureStatus::Invalid:
    return "invalid";
  case ClosureStatus::Failed:
    return "failed";
  case ClosureStatus::ArithmeticFailure:
    return "arithmetic_failure";
  case ClosureStatus::RouteFailure:
    return "route_failure";
  case ClosureStatus::RouteTerminalMismatch:
    return "route_terminal_mismatch";
  case ClosureStatus::MappingNonclosure:
    return "mapping_nonclosure";
  case ClosureStatus::SelectedHandshakeCycle:
    return "selected_handshake_cycle";
  case ClosureStatus::Closed:
    return "closed";
  case ClosureStatus::TemporaryMapping:
    return "temporary_mapping";
  case ClosureStatus::FixedTerminalCutTemporary:
    return "fixed_terminal_cut_temporary";
  case ClosureStatus::FixedTerminalCut:
    return "fixed_terminal_cut";
  case ClosureStatus::NoProgressTemporary:
    return "no_progress_temporary";
  case ClosureStatus::NoProgress:
    return "no_progress";
  case ClosureStatus::TemporaryCapacity:
    return "temporary_capacity";
  case ClosureStatus::IterationLimit:
    return "iteration_limit";
  }
  llvm_unreachable("unknown Mapping closure status");
}

void emit(Level minimum, Stage stage, Event event,
          llvm::function_ref<void(llvm::json::Object &)> buildFields) {
  emitInvocationDiagnostic(minimum, stage, event, [&] {
    llvm::json::Object payload;
    if (buildFields)
      buildFields(payload);
    return llvm::json::Value(std::move(payload));
  });
}

void MappingRunStatistics::emit(Stage stage,
                                ClosureStatus closureStatus) const {
  emit(stage, closureStatus, {});
}

void MappingRunStatistics::emit(
    Stage stage, ClosureStatus closureStatus,
    llvm::function_ref<void(llvm::json::Object &)> buildFields) const {
  emit(Level::Summary, stage, closureStatus, buildFields);
}

void MappingRunStatistics::emit(
    Level minimum, Stage stage, ClosureStatus closureStatus,
    llvm::function_ref<void(llvm::json::Object &)> buildFields) const {
  mapping_debug::emit(minimum, stage, Event::Statistics,
                      [&](llvm::json::Object &fields) {
                        fields["candidate_rows"] = candidateRows;
                        fields["candidate_publications"] =
                            candidatePublications;
                        fields["actions_proposed"] = actionsProposed;
                        fields["actions_accepted"] = actionsAccepted;
                        fields["actions_rejected"] = actionsRejected;
                        fields["actions_rolled_back"] = actionsRolledBack;
                        fields["a_star_expansions"] = aStarExpansions;
                        fields["negotiated_iterations"] = negotiatedIterations;
                        fields["capacity_conflicts"] = capacityConflicts;
                        fields["arithmetic_failures"] = arithmeticFailures;
                        fields["closure_status"] =
                            closureStatusSpelling(closureStatus);
                        if (buildFields)
                          buildFields(fields);
                      });
}

} // namespace loom::mapping_debug
