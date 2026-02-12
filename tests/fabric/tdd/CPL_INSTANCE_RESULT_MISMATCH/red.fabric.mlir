// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_RESULT_MISMATCH

// @inc returns one i32 result, but the instance declares two results.
fabric.module @inc(%a: i32) -> (i32) {
  %r = fabric.pe %a : (i32) -> (i32) {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %s = arith.addi %x, %c1 : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : i32
}

fabric.module @top(%v: i32) -> (i32, i32) {
  %o0, %o1 = fabric.instance @inc(%v) : (i32) -> (i32, i32)
  fabric.yield %o0, %o1 : i32, i32
}
