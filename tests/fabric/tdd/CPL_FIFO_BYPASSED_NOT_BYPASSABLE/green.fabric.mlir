// RUN: loom --adg %s

// Valid: bypassable fifo with bypassed = false.
fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 2, bypassable] {bypassed = false} %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
