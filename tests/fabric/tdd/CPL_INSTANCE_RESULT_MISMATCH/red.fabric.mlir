// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_INSTANCE_RESULT_MISMATCH

// @inc returns one i32 result, but the instance declares two results.
fabric.module @inc(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a : (!dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %s = arith.addi %x, %c1 : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}

fabric.module @top(%v: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o0, %o1 = fabric.instance @inc(%v) : (!dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  fabric.yield %o0, %o1 : !dataflow.bits<32>, !dataflow.bits<32>
}
