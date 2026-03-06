// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MODULE_PORT_ORDER

// Incorrect port ordering: tagged input appears before native input.
fabric.module @order_bad(%a: !dataflow.tagged<!dataflow.bits<32>, i4>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %sum = fabric.pe %b, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : !dataflow.bits<32>
}
