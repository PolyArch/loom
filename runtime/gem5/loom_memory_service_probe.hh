#ifndef LOOM_RUNTIME_GEM5_MEMORY_SERVICE_PROBE_HH
#define LOOM_RUNTIME_GEM5_MEMORY_SERVICE_PROBE_HH

#include "base/addr_range.hh"
#include "mem/probes/base.hh"
#include "params/LoomMemoryServiceProbe.hh"
#include <cstdint>
#include <vector>

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
  bool isConfigurationTransport(Addr address) const;
  System *const system;
  const Tick serviceTicksPerByte;
  /// Guest apertures holding the immutable binary configuration image. A
  /// SpatialCore streams that image from here before its first launch.
  const std::vector<AddrRange> configurationTransportRanges;
  bool active = false;
  /// End of the service the memory has already accepted, configuration
  /// transport included, so overlapping acceptance remains detectable.
  Tick serviceEnd = 0;
  /// End of the last accepted application-data service, which is the only
  /// service the integral below counts.
  Tick applicationServiceEnd = 0;
  std::uint64_t scheduledServiceTicks = 0;
};

} // namespace gem5
#endif
