// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_SWITCH_ROUTE_LEN_MISMATCH

// connectivity_table has 2 ones so route_table must have length 2, but 3 given.
fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] {route_table = [1, 0, 1]} %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o1, %o2 : !dataflow.bits<32>, !dataflow.bits<32>
}
