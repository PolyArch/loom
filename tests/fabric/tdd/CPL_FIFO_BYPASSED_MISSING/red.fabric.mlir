// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FIFO_BYPASSED_MISSING

// Invalid: bypassable without bypassed attribute.
fabric.module @test(%a: i32) -> (i32) {
  %out = fabric.fifo [depth = 2, bypassable] %a : i32
  fabric.yield %out : i32
}
