from m5.objects.Device import DmaDevice
from m5.params import *
from m5.proxy import *


class LoomDmaSpikeDevice(DmaDevice):
    type = "LoomDmaSpikeDevice"
    cxx_class = "gem5::LoomDmaSpikeDevice"
    cxx_header = "dev/loom/LoomDmaSpikeDevice.hh"

    pio_addr = Param.Addr("Device Address")
    pio_latency = Param.Latency("100ns", "Programmed IO latency")
    pio_size = Param.Addr(0x1000, "MMIO aperture size")
