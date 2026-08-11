from m5.objects.RiscvFsWorkload import RiscvBareMetal
from m5.params import Param, VectorParam


class LoomRiscvDeploymentWorkload(RiscvBareMetal):
    type = "LoomRiscvDeploymentWorkload"
    cxx_class = "gem5::LoomRiscvDeploymentWorkload"
    cxx_header = "runtime/gem5/loom_riscv_deployment_workload.hh"

    host_cpu_id = Param.Unsigned("HostCore gem5 CPU identifier")
    host_entry_symbol = Param.String("Deployment-selected host entry symbol")
    host_dispatch_address = Param.Addr("Thread Dispatch MMIO base address")
    stack_base = Param.Addr("Lowest per-CPU stack address")
    stack_stride = Param.Addr("Per-CPU stack allocation size")
    instruction_images = VectorParam.String(
        [], "Exact InstructionCore ELF images"
    )
    runtime_images = VectorParam.String([], "Deployment runtime image files")
    runtime_image_addresses = VectorParam.Addr(
        [], "Physical addresses for Deployment runtime images"
    )
    target_cpu_ids = VectorParam.Unsigned(
        [], "CPU identifier for each Thread Dispatch target"
    )
    target_image_ordinals = VectorParam.Unsigned(
        [], "Instruction image ordinal for each dispatch target"
    )
    target_entry_symbols = VectorParam.String(
        [], "Instruction entry symbol for each dispatch target"
    )
    target_bridge_addresses = VectorParam.Addr(
        [], "Spatial Bridge MMIO address for each dispatch target"
    )
    target_launch_addresses = VectorParam.Addr(
        [], "Spatial launch payload address for each dispatch target"
    )
    target_launch_sizes = VectorParam.Unsigned(
        [], "Spatial launch payload size for each dispatch target"
    )
