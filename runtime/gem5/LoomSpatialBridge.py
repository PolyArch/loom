from m5.objects.Device import DmaDevice
from m5.params import NULL, Param


class LoomSpatialBridge(DmaDevice):
    type = "LoomSpatialBridge"
    cxx_class = "gem5::LoomSpatialBridge"
    cxx_header = "runtime/gem5/loom_spatial_bridge.hh"

    pio_addr = Param.Addr("Bridge MMIO base address")
    pio_latency = Param.Latency("100ns", "Bridge MMIO access latency")
    pio_size = Param.Addr(0x1000, "Bridge MMIO aperture size")
    session_ordinal = Param.Unsigned("System bridge session ordinal")
    engine_session = Param.LoomSpatialEngineSession(NULL, "Shared Spatial engine session; absent for an idle bridge")
    result_path = Param.String("Normalized bridge result destination")
    max_message_bytes = Param.Unsigned(
        64 * 1024 * 1024, "Maximum accepted bridge message size"
    )
    max_invocations = Param.Unsigned(4096, "Maximum session invocation count")
    collect_performance = Param.Bool(
        False, "Collect attempt-local host performance observations"
    )
