#ifndef LOOM_COMMON_MAPPINGDEBUGLOG_H
#define LOOM_COMMON_MAPPINGDEBUGLOG_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>

namespace loom::mapping_debug {

enum class Level : std::uint8_t {
  Disabled = 0,
  Summary = 1,
  Decision = 2,
  Detail = 3,
};

enum class Stage : std::uint8_t {
  TechMapping,
  SpatialPnr,
  SystemPnr,
};

enum class Event : std::uint8_t {
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
  ArithmeticFailure,
  MappingFailure,
};

/// Returns the process-wide level parsed once from LOOM_DEBUG_VERBOSE.
Level level();

bool enabled(Level minimum);

/// Emits one line-atomic JSON event. The field builder is not invoked when the
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

  void emit(Stage stage, llvm::StringRef closureStatus) const;
};

} // namespace loom::mapping_debug

#endif // LOOM_COMMON_MAPPINGDEBUGLOG_H
