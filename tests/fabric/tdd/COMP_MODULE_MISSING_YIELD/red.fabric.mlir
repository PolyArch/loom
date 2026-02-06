// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MODULE_MISSING_YIELD

// The module declares one result (i32) but yield has zero operands.
fabric.module @yield_bad(%a: i32, %b: i32) -> (i32) {
  %sum = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield
}
