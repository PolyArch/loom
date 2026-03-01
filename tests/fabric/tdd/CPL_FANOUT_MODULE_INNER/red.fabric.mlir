// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FANOUT_MODULE_INNER

// An instance output is used by two consumers (switch + yield) without
// switch broadcast. This must be rejected.
fabric.pe @add(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: !dataflow.bits<32>, %y: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %sum = fabric.instance @add(%x, %y) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  // %sum used twice: feeds both yield operands
  fabric.yield %sum, %sum : !dataflow.bits<32>, !dataflow.bits<32>
}
