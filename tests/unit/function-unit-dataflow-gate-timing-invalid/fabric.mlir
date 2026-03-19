module {
  fabric.spatial_pe @bad_pe(%value: index, %cond: i1) -> (index, i1) {
    fabric.function_unit @fu_bad(%arg0: index, %arg1: i1) -> (index, i1)
        [latency = 1, interval = 1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : index, i1 -> index, i1
      fabric.yield %0, %1 : index, i1
    }
    fabric.yield
  }

  fabric.module @function_unit_dataflow_gate_timing_invalid(%value: index, %cond: i1)
      -> (index, i1) {
    %out:2 = fabric.instance @bad_pe(%value, %cond) {sym_name = "pe_bad"}
        : (index, i1) -> (index, i1)
    fabric.yield %out#0, %out#1 : index, i1
  }
}
