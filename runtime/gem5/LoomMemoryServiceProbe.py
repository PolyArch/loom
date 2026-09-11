from m5.objects.BaseMemProbe import BaseMemProbe
from m5.params import Param, VectorParam
from m5.proxy import Parent
from m5.util.pybind import PyBindMethod


class LoomMemoryServiceProbe(BaseMemProbe):
    type = "LoomMemoryServiceProbe"
    cxx_class = "gem5::LoomMemoryServiceProbe"
    cxx_header = "runtime/gem5/loom_memory_service_probe.hh"
    cxx_exports = [PyBindMethod("beginWindow"), PyBindMethod("occupiedTicks")]

    system = Param.System(Parent.any, "Exact timing-mode System being observed")
    service_ticks_per_byte = Param.Tick("Exact SimpleMemory acceptance service cost")
    configuration_transport_ranges = VectorParam.AddrRange(
        [], "Guest apertures holding the immutable binary configuration image"
    )
