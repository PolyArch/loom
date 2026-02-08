// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_FIFO_BYPASSED_NOT_BYPASSABLE

// Invalid: bypassed set without bypassable.
fabric.module @test(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 2] {bypassed = true} %a : i32
  fabric.yield %out : i32
}
