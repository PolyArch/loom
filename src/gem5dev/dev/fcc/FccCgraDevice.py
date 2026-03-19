from m5.objects.Device import BasicPioDevice
from m5.params import *
from m5.proxy import *


class FccCgraDevice(BasicPioDevice):
    type = "FccCgraDevice"
    cxx_class = "gem5::FccCgraDevice"
    cxx_header = "dev/fcc/FccCgraDevice.hh"

    pio_size = Param.Addr(0x1000, "MMIO aperture size")
    runtime_manifest = Param.String("", "Path to FCC runtime manifest")
    fcc_binary = Param.String("", "Path to fcc replay binary")
    bridge_script = Param.String("", "Path to the gem5-to-fcc bridge script")
    work_dir = Param.String("", "Per-run scratch directory for accelerator replay")
