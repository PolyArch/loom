#ifndef LOOM_SIMULATOR_SPATIALEXECUTIONSESSION_H
#define LOOM_SIMULATOR_SPATIALEXECUTIONSESSION_H

#include "Simulator/SimulationExecution.h"

#include <cstdint>

namespace loom::sim {

/// Ephemeral control state shared by Spatial execution providers. Artifact
/// terminals remain owned by SimulationExecution; this state only controls a
/// live, pausable attempt.
enum class SpatialExecutionSessionState {
  Runnable,
  Retired,
  Halted,
  StoppedByLimit,
  Failed,
  WaitingForExternalMemory,
  WaitingForExternalStreamInput,
};

/// One demanded event at a live graph stream input. The occurrence is the
/// next dense input-token ordinal of this activation; it is not a producer
/// activation segment or an ordered-channel sequence number.
struct SpatialStreamInputRequest final {
  std::uint64_t streamInputOrdinal;
  std::uint64_t occurrenceOrdinal;
  SpatialEventCoordinate readyCoordinate;
};

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALEXECUTIONSESSION_H
