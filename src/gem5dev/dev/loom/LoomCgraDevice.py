from m5.objects.Device import DmaDevice
from m5.params import *
from m5.proxy import *


class LoomCgraDevice(DmaDevice):
    type = "LoomCgraDevice"
    cxx_class = "gem5::LoomCgraDevice"
    cxx_header = "dev/loom/LoomCgraDevice.hh"

    pio_addr = Param.Addr("Device Address")
    pio_latency = Param.Latency("100ns", "Programmed IO latency")
    pio_size = Param.Addr(0x1000, "MMIO aperture size")
    accel_cycle_latency = Param.Latency("1ns", "Accelerator cycle latency")
    max_batch_cycles = Param.Unsigned(
        1024, "Maximum accelerator cycles executed per gem5 event"
    )
    max_inflight_mem_reqs = Param.Unsigned(
        8, "Maximum external memory DMA requests in flight"
    )
    sim_image = Param.String("", "Path to LOOM simulator runtime image")
    runtime_manifest = Param.String("", "Path to LOOM runtime manifest")
    work_dir = Param.String("", "Per-run scratch directory for accelerator artifacts")
