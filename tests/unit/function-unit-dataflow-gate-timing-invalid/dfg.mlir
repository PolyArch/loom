module {
  handshake.func @function_unit_dataflow_gate_timing_invalid(%value: index, %cond: i1)
      -> (index, i1) {
    %0, %1 = dataflow.gate %value, %cond : index, i1 -> index, i1
    handshake.return %0, %1 : index, i1
  }
}
