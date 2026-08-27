from m5.objects.Device import BasicPioDevice
from m5.params import Param, VectorParam


class LoomThreadDispatch(BasicPioDevice):
    type = "LoomThreadDispatch"
    cxx_class = "gem5::LoomThreadDispatch"
    cxx_header = "runtime/gem5/loom_thread_dispatch.hh"

    workload = Param.LoomRiscvDeploymentWorkload(
        "Deployment workload that owns exact executable entries"
    )
    root_event_trace_path = Param.String(
        "Canonical root lifecycle attempt output"
    )
    root_event_control_path = Param.String(
        "", "Optional acknowledged root lifecycle control socket"
    )
    logical_target_count = Param.UInt64(
        "Number of logical dispatch targets in each endpoint"
    )
    endpoint_target_offsets = VectorParam.UInt64(
        "Physical target offset selected by each runtime endpoint"
    )
    endpoint_dispatch_enabled = VectorParam.UInt64(
        "Whether each runtime endpoint admits a later dispatch"
    )
