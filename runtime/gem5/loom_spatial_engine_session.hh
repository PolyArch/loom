#ifndef LOOM_RUNTIME_GEM5_LOOM_SPATIAL_ENGINE_SESSION_HH
#define LOOM_RUNTIME_GEM5_LOOM_SPATIAL_ENGINE_SESSION_HH

#include "Runtime/Gem5BridgeWire.h"

#include "params/LoomSpatialEngineSession.hh"
#include "sim/sim_object.hh"

#include <deque>
#include <map>
#include <string>

namespace gem5 {

class LoomSpatialBridge;

// Owns the causal frontier of one external engine, including every bridge
// whose channel state that engine can wake. Socket waits occur only in advance,
// after the simulation loop has returned to its driver at the input tick.
class LoomSpatialEngineSession final : public SimObject {
public:
  using Params = LoomSpatialEngineSessionParams;
  explicit LoomSpatialEngineSession(const Params &params);
  ~LoomSpatialEngineSession() override;

  void registerBridge(std::uint64_t ordinal, LoomSpatialBridge &bridge);
  void submit(loom::runtime::Gem5BridgeMessage message);
  void advance();
  bool isAdvanceExit(const std::string &cause) const;

private:
  const std::string socketPath;
  const std::uint64_t bridgeCount;
  int socket = -1;
  std::uint64_t generation = 0;
  std::map<std::uint64_t, LoomSpatialBridge *> bridges;
  std::deque<loom::runtime::Gem5BridgeAdvance> pending;

  void connectEngine();
};

} // namespace gem5

#endif // LOOM_RUNTIME_GEM5_LOOM_SPATIAL_ENGINE_SESSION_HH
