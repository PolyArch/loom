// RUN: loom --adg %s

// FIFO with a valid tagged type.
fabric.module @test_fifo_tagged(%a: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %out = fabric.fifo [depth = 2] %a : !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i4>
}
