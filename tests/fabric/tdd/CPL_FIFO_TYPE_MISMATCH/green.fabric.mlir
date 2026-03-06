// RUN: loom --adg %s

// Valid FIFO with matching input/output types.
fabric.module @test_fifo_type_ok(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 4] %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
