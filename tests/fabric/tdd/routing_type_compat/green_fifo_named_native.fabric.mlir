// RUN: loom --adg %s

// Named FIFO with bit-width-compatible native types: i32 in, f32 out.
fabric.fifo @relaxed_buf [depth = 4] : (!dataflow.bits<32>) -> (!dataflow.bits<32>)

fabric.module @test_fifo_named_compat(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 4] %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
