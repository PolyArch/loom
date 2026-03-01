// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_BYPASSED_NOT_BYPASSABLE

// Invalid: bypassed set without bypassable.
fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 2] {bypassed = true} %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
