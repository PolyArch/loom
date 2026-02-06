// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MODULE_PORT_ORDER

// Incorrect port ordering: tagged input appears before native input.
fabric.module @order_bad(%a: !dataflow.tagged<i32, i4>, %b: i32) -> (i32) {
  %sum = fabric.pe %b, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}
