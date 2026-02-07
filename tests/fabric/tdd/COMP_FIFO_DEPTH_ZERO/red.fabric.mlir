// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_FIFO_DEPTH_ZERO

// FIFO with depth = 0 is not allowed.
fabric.module @test_fifo_depth_zero(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 0] %a : i32
  fabric.yield %out : i32
}
