// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_BYPASSED_MISSING

// Invalid: bypassable without bypassed attribute.
fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 2, bypassable] %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
