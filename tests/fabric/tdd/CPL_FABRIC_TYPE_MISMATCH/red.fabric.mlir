// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FABRIC_TYPE_MISMATCH

// The module declares i32 as its result type, but the yield operand is i64.
fabric.module @type_bad(%a: !dataflow.bits<64>, %b: !dataflow.bits<64>) -> (!dataflow.bits<32>) {
  %sum = fabric.pe %a, %b : (!dataflow.bits<64>, !dataflow.bits<64>) -> (!dataflow.bits<64>) {
  ^bb0(%x: i64, %y: i64):
    %r = arith.addi %x, %y : i64
    fabric.yield %r : i64
  }
  fabric.yield %sum : !dataflow.bits<64>
}
