// Negative test: dataflow.stream FU with latency=3 instead of latency=-1.
// dataflow.stream is a stateful FU that requires latency=-1 (variable);
// SVGen must reject any explicit positive latency for this op.
module {
  fabric.spatial_pe @pe_def(%p0: index, %p1: index, %p2: index)
      -> (index, i1) {
    fabric.function_unit @fu_stream(%arg0: index, %arg1: index, %arg2: index)
        -> (index, i1) [latency = 3, interval = 1] {
      %idx, %cont = dataflow.stream %arg0, %arg1, %arg2
          {step_op = "+=", cont_cond = "<"} : (index, index, index) -> (index, i1)
      fabric.yield %idx, %cont : index, i1
    }
    fabric.yield
  }

  fabric.module @test_dataflow_wrong_latency(
      %a: index, %b: index, %c: index)
      -> (index, i1) {
    %out:2 = fabric.instance @pe_def(%a, %b, %c) {sym_name = "pe_0"}
        : (index, index, index) -> (index, i1)
    fabric.yield %out#0, %out#1 : index, i1
  }
}
