// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_FANOUT_MODULE_BOUNDARY

// A module input argument feeds two instance input ports without
// switch broadcast. This must be rejected.
fabric.pe @add(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  // %x used twice: feeds both inputs of @add
  %sum = fabric.instance @add(%x, %x) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %sum : !dataflow.bits<32>
}
