// RUN: loom --adg %s

// FIFO with bits<16> passes verification.
fabric.module @test_fifo_bits(%a: !dataflow.bits<16>) -> (!dataflow.bits<16>) {
  %out = fabric.fifo [depth = 4] %a : !dataflow.bits<16>
  fabric.yield %out : !dataflow.bits<16>
}
