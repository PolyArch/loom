from m5.objects.Device import DmaDevice
from m5.params import Param


class LoomSpatialBridge(DmaDevice):
    type = "LoomSpatialBridge"
    cxx_class = "gem5::LoomSpatialBridge"
    cxx_header = "runtime/gem5/loom_spatial_bridge.hh"

    pio_addr = Param.Addr("Bridge MMIO base address")
    pio_latency = Param.Latency("100ns", "Bridge MMIO access latency")
    pio_size = Param.Addr(0x1000, "Bridge MMIO aperture size")
    engine_socket = Param.String("Invocation-local Spatial engine socket")
    launch_payload = Param.String("Canonical Spatial launch payload file")
    result_path = Param.String("Normalized bridge result destination")
    max_message_bytes = Param.Unsigned(
        64 * 1024 * 1024, "Maximum accepted bridge message size"
    )
