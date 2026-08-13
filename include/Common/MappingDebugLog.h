#ifndef LOOM_COMMON_MAPPINGDEBUGLOG_H
#define LOOM_COMMON_MAPPINGDEBUGLOG_H

#include "Common/DiagnosticVerbosity.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>

namespace loom::mapping_debug {

using Level = DiagnosticVerbosity;

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
  TopologyQuality,
  TagDomainPressure,
  ArithmeticFailure,
  MappingFailure,
};

/// Returns the Common-owned process-wide diagnostic level.
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
  void emit(Stage stage, llvm::StringRef closureStatus,
            llvm::function_ref<void(llvm::json::Object &)> buildFields) const;
};

} // namespace loom::mapping_debug

#endif // LOOM_COMMON_MAPPINGDEBUGLOG_H
