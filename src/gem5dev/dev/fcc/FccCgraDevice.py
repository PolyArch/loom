from m5.objects.Device import DmaDevice
from m5.params import *
from m5.proxy import *


class FccCgraDevice(DmaDevice):
    type = "FccCgraDevice"
    cxx_class = "gem5::FccCgraDevice"
    cxx_header = "dev/fcc/FccCgraDevice.hh"

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
    sim_image = Param.String("", "Path to FCC simulator runtime image")
    runtime_manifest = Param.String("", "Path to FCC runtime manifest")
    work_dir = Param.String("", "Per-run scratch directory for accelerator artifacts")
