#include "runtime/gem5/loom_memory_service_probe.hh"

#include "base/logging.hh"
#include "sim/cur_tick.hh"
#include "sim/system.hh"
#include <limits>

namespace gem5 {

LoomMemoryServiceProbe::LoomMemoryServiceProbe(
    const LoomMemoryServiceProbeParams &params)
    : BaseMemProbe(params), system(params.system),
      serviceTicksPerByte(params.service_ticks_per_byte),
      configurationTransportRanges(params.configuration_transport_ranges) {
  panic_if(serviceTicksPerByte == 0, "SimpleMemory service cost must be positive");
}

void LoomMemoryServiceProbe::beginWindow() {
  panic_if(active, "memory service observation window already started");
  panic_if(!system || !system->isTimingMode(),
           "memory service observation requires the exact timing-mode System");
  active = true;
  serviceEnd = curTick();
  applicationServiceEnd = curTick();
}

bool LoomMemoryServiceProbe::isConfigurationTransport(Addr address) const {
  for (const AddrRange &range : configurationTransportRanges)
    if (range.contains(address))
      return true;
  return false;
}

void LoomMemoryServiceProbe::handleRequest(const probing::PacketInfo &packet) {
  if (!active)
    return;
  panic_if(!system->isTimingMode(), "memory service observation left timing mode");
  // The fixed timing-CPU and timing-DMA topology never uses recvAtomic transport.
  // CommMonitor notifies only after SimpleMemory accepts the request. Swaps
  // and failed store-conditionals consume this same service despite possibly
  // changing neither AbstractMemory's read-byte nor write-byte counter.
  panic_if(!(packet.cmd.isRead() || packet.cmd.isWrite()),
           "SimpleMemory accepted a non-memory-service request");
  panic_if(curTick() < serviceEnd,
           "SimpleMemory accepted overlapping service intervals");
  panic_if(packet.size > std::numeric_limits<Tick>::max() / serviceTicksPerByte,
           "memory service duration overflow");
  const Tick duration = packet.size * serviceTicksPerByte;
  panic_if(duration > std::numeric_limits<Tick>::max() - curTick() ||
               duration > std::numeric_limits<std::uint64_t>::max() - scheduledServiceTicks,
           "memory service observation overflow");
  serviceEnd = curTick() + duration;
  // Configuration transport streams the immutable binary configuration image
  // into a SpatialCore. It occupies the memory, but it is launch overhead
  // rather than application data movement: the accelerated window charges it
  // to its configuration residency phase, so it never inflates the service
  // occupancy that the saturation branches divide by their phase.
  if (isConfigurationTransport(packet.addr))
    return;
  applicationServiceEnd = serviceEnd;
  scheduledServiceTicks += duration;
}

std::uint64_t LoomMemoryServiceProbe::occupiedTicks() const {
  panic_if(!active, "memory service observation window is absent");
  const Tick pending =
      applicationServiceEnd > curTick() ? applicationServiceEnd - curTick() : 0;
  return scheduledServiceTicks - pending;
}

} // namespace gem5
