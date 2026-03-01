// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MODULE_MISSING_YIELD

// The module declares one result (i32) but yield has zero operands.
fabric.module @yield_bad(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %sum = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield
}
