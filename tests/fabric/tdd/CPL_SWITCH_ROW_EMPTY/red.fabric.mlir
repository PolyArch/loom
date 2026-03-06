// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_SWITCH_ROW_EMPTY

// Output row 1 (second row) has no connections (all zeros).
fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 1, 0, 0]] %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o1, %o2 : !dataflow.bits<32>, !dataflow.bits<32>
}
