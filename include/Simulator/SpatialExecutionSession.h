#ifndef LOOM_SIMULATOR_SPATIALEXECUTIONSESSION_H
#define LOOM_SIMULATOR_SPATIALEXECUTIONSESSION_H

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
};

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALEXECUTIONSESSION_H
