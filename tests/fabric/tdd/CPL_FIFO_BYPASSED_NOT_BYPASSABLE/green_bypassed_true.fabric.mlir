// RUN: loom --adg %s

// Valid: bypassable fifo with bypassed = true.
fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 4, bypassable] {bypassed = true} %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
