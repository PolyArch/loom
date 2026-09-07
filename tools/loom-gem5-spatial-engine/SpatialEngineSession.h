#ifndef LOOM_TOOLS_GEM5_SPATIAL_ENGINE_SESSION_H
#define LOOM_TOOLS_GEM5_SPATIAL_ENGINE_SESSION_H

#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5SpatialChannel.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"

#include <memory>
#include <optional>
#include <vector>

namespace loom::gem5engine {

llvm::Error invalid(const llvm::Twine &message);

#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
using PreparedSpatialExecution = loom::sim::PreparedDfgExecution;
#else
using PreparedSpatialExecution = loom::sim::PreparedCgraExecution;
#endif

struct SpatialSessionEntry final {
  loom::sim::ImportedSpatialSimulationWorkload workload;
  std::optional<loom::sim::CanonicalSimulationRuntimeInput> staticRuntime;
  loom::runtime::Gem5SpatialChannelProjection projection;
  std::vector<std::uint8_t> expectedLaunch;
  std::uint64_t bridgeOrdinal = 0;
  std::uint64_t sessionEntryOrdinal = 0;
  std::size_t preparedOrdinal = 0;
};

struct SpatialEngineLimits final {
  std::uint64_t maximumWork;
  std::uint64_t ticksPerCycle;
  std::uint64_t maximumInvocations;
};

class SpatialEngineSession final {
public:
  static llvm::Expected<std::unique_ptr<SpatialEngineSession>>
  create(std::vector<SpatialSessionEntry> entries,
         std::vector<PreparedSpatialExecution> preparedExecutions,
         SpatialEngineLimits limits, std::string performanceProfilePath);
  ~SpatialEngineSession();

  llvm::Expected<loom::runtime::Gem5BridgeAdvance>
  advance(const loom::runtime::Gem5BridgeAdvance &input);

private:
  struct Impl;
  explicit SpatialEngineSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace loom::gem5engine

#endif // LOOM_TOOLS_GEM5_SPATIAL_ENGINE_SESSION_H
