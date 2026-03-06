// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_SWITCH_COL_EMPTY

// Input column 1 (second input) has no connections in any output row.
fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 1, 0]] %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o1, %o2 : !dataflow.bits<32>, !dataflow.bits<32>
}
