// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_DEPTH_ZERO

// FIFO with depth = 0 is not allowed.
fabric.module @test_fifo_depth_zero(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.fifo [depth = 0] %a : !dataflow.bits<32>
  fabric.yield %out : !dataflow.bits<32>
}
