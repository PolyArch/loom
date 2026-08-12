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

void emit(Level minimum, Stage stage, Event event,
          llvm::function_ref<void(llvm::json::Object &)> buildFields) {
  if (!enabled(minimum))
    return;

  llvm::json::Object fields;
  if (buildFields)
    buildFields(fields);

  OutputState &state = outputState();
  std::lock_guard<std::mutex> lock(state.mutex);
  fields["schema"] = "loom.mapping.debug.1";
  fields["level"] = static_cast<std::int64_t>(minimum);
  fields["event"] = spelling(event);
  fields["stage"] = spelling(stage);
  fields["sequence"] = static_cast<std::int64_t>(state.nextSequence++);

  std::string line;
  llvm::raw_string_ostream stream(line);
  stream << llvm::json::Value(std::move(fields));
  stream.flush();
  llvm::errs() << line << '\n';
}

void MappingRunStatistics::emit(Stage stage,
                                llvm::StringRef closureStatus) const {
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
                        fields["closure_status"] = closureStatus;
                      });
}

} // namespace loom::mapping_debug
