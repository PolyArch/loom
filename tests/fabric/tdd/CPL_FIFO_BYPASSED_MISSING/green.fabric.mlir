// RUN: loom --adg %s

// Valid: non-bypassable fifo without bypassed.
fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 2] %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
