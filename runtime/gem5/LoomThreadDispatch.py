from m5.objects.Device import BasicPioDevice
from m5.params import Param


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
