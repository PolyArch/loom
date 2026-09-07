from m5.SimObject import SimObject
from m5.params import Param
from m5.util.pybind import PyBindMethod


class LoomSpatialEngineSession(SimObject):
    type = "LoomSpatialEngineSession"
    cxx_class = "gem5::LoomSpatialEngineSession"
    cxx_header = "runtime/gem5/loom_spatial_engine_session.hh"
    cxx_exports = [PyBindMethod("advance"), PyBindMethod("isAdvanceExit")]

    engine_socket = Param.String("Invocation-local Spatial engine socket")
    bridge_count = Param.Unsigned("Number of bridges sharing this engine")
