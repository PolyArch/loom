// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_PE_EMPTY_BODY

// A fabric.pe whose body contains only the terminator (no non-terminator ops).
fabric.module @test(%a: i32) -> (i32) {
  %r = fabric.pe %a : (i32) -> (i32) {
  ^bb0(%x: i32):
    fabric.yield %x : i32
  }
  fabric.yield %r : i32
}
