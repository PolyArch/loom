// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_FABRIC_TYPE_MISMATCH

// The module declares i32 as its result type, but the yield operand is i64.
fabric.module @type_bad(%a: i64, %b: i64) -> (i32) {
  %sum = fabric.pe %a, %b : (i64, i64) -> (i64) {
  ^bb0(%x: i64, %y: i64):
    %r = arith.addi %x, %y : i64
    fabric.yield %r : i64
  }
  fabric.yield %sum : i64
}
