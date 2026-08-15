#include "Common/MappingDebugLog.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <string>

namespace loom::mapping_debug {
namespace {

llvm::StringRef spelling(Stage stage) {
  switch (stage) {
  case Stage::TechMapping:
    return "tech_mapping";
  case Stage::SpatialPnr:
    return "spatial_pnr";
  case Stage::SystemPnr:
    return "system_pnr";
  }
  llvm_unreachable("unknown Mapping debug stage");
}

llvm::StringRef spelling(Event event) {
  switch (event) {
  case Event::InvocationBegin:
    return "invocation_begin";
  case Event::InvocationEnd:
    return "invocation_end";
  case Event::Statistics:
    return "statistics";
  case Event::Candidate:
    return "candidate";
  case Event::Seed:
    return "seed";
  case Event::NegotiationIteration:
    return "negotiation_iteration";
  case Event::CapacityConflict:
    return "capacity_conflict";
  case Event::ActionProposal:
    return "action_proposal";
  case Event::ActionOutcome:
    return "action_outcome";
  case Event::ContextChoice:
    return "context_choice";
  case Event::NetRoute:
    return "net_route";
  case Event::CutAnalysis:
    return "cut_analysis";
  case Event::TopologyQuality:
    return "topology_quality";
  case Event::TagDomainPressure:
    return "tag_domain_pressure";
  case Event::ArithmeticFailure:
    return "arithmetic_failure";
  case Event::MappingFailure:
    return "mapping_failure";
  }
  llvm_unreachable("unknown Mapping debug event");
}

struct OutputState final {
  std::mutex mutex;
  std::uint64_t nextSequence = 0;
};

OutputState &outputState() {
  static OutputState state;
  return state;
}

} // namespace

Level level() { return diagnosticVerbosity(); }

bool enabled(Level minimum) { return diagnosticVerbosityEnabled(minimum); }

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
  if (!enabled(minimum))
    return;

  llvm::json::Object payload;
  if (buildFields)
    buildFields(payload);

  OutputState &state = outputState();
  std::lock_guard<std::mutex> lock(state.mutex);
  llvm::json::Object envelope;
  envelope["schema"] = "loom.mapping.debug.1";
  envelope["level"] = static_cast<std::int64_t>(minimum);
  envelope["event"] = spelling(event);
  envelope["stage"] = spelling(stage);
  envelope["sequence"] = static_cast<std::int64_t>(state.nextSequence++);
  envelope["payload"] = std::move(payload);

  std::string line;
  llvm::raw_string_ostream stream(line);
  stream << llvm::json::Value(std::move(envelope));
  stream.flush();
  llvm::errs() << line << '\n';
}

void MappingRunStatistics::emit(Stage stage,
                                ClosureStatus closureStatus) const {
  emit(stage, closureStatus, {});
}

void MappingRunStatistics::emit(
    Stage stage, ClosureStatus closureStatus,
    llvm::function_ref<void(llvm::json::Object &)> buildFields) const {
  mapping_debug::emit(Level::Summary, stage, Event::Statistics,
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
