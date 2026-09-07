#ifndef LOOM_RUNTIME_GEM5_MEMORY_SERVICE_PROBE_HH
#define LOOM_RUNTIME_GEM5_MEMORY_SERVICE_PROBE_HH

#include "mem/probes/base.hh"
#include "params/LoomMemoryServiceProbe.hh"
#include <cstdint>

namespace gem5 {
class System;

/// Observes CommMonitor's accepted downstream requests immediately before the
/// sole SimpleMemory. No packet forwarding, scheduling, or simulator state changes.
class LoomMemoryServiceProbe final : public BaseMemProbe {
public:
  explicit LoomMemoryServiceProbe(const LoomMemoryServiceProbeParams &params);
  void beginWindow();
  std::uint64_t occupiedTicks() const;

private:
  void handleRequest(const probing::PacketInfo &packet) override;
  System *const system;
  const Tick serviceTicksPerByte;
  bool active = false;
  Tick serviceEnd = 0;
  std::uint64_t scheduledServiceTicks = 0;
};

} // namespace gem5
#endif
