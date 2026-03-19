module {
  fabric.spatial_pe @stream_pe(%start: index, %step: index, %bound: index)
      -> (index, i1) {
    fabric.function_unit @fu_stream(%arg0: index, %arg1: index, %arg2: index)
        -> (index, i1) [latency = 1, interval = 1] {
      %idx, %cont = dataflow.stream %arg0, %arg1, %arg2
          {step_op = "+=", cont_cond = "<"} : (index, index, index) -> (index, i1)
      fabric.yield %idx, %cont : index, i1
    }
    fabric.yield
  }

  fabric.module @function_unit_stream_timing_invalid(
      %start: index, %step: index, %bound: index)
      -> (index, i1) {
    %out:2 = fabric.instance @stream_pe(%start, %step, %bound)
        {sym_name = "pe_stream"} : (index, index, index) -> (index, i1)
    fabric.yield %out#0, %out#1 : index, i1
  }
}
