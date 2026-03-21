// Negative test: dataflow.stream FU with latency=3 instead of latency=-1.
// dataflow.stream is a stateful FU that requires latency=-1 (variable);
// SVGen must reject any explicit positive latency for this op.
module {
  fabric.function_unit @fu_stream(%start: index, %step: index, %bound: index)
      -> (index, i1)
      [latency = 3, interval = 1] {
    %idx, %cont = dataflow.stream(%start, %step, %bound)
        : (index, index, index) -> (index, i1)
    fabric.yield %idx, %cont : index, i1
  }

  fabric.spatial_pe @pe_def(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>,
                            %p2: !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.instance @fu_stream() {sym_name = "fu_stream_0"} : () -> ()
    fabric.yield
  }

  fabric.module @test_dataflow_wrong_latency(
      %a: !fabric.bits<32>, %b: !fabric.bits<32>, %c: !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>) {
    %out:2 = fabric.instance @pe_def(%a, %b, %c) {sym_name = "pe_0"}
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %out#0, %out#1 : !fabric.bits<32>, !fabric.bits<32>
  }
}
