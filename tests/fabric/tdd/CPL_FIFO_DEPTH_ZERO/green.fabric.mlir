// RUN: loom --adg %s

// FIFO with depth >= 1 is valid.
fabric.module @test_fifo_depth_ok(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 1] %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
