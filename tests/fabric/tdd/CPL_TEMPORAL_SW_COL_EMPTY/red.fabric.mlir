// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_SW_COL_EMPTY

// Input column 1 has no connections (all zeros).
// 2 outputs, 2 inputs: row 0 = [1, 0], row 1 = [1, 0].
fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>, %b: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 2, connectivity_table = [1, 0, 1, 0]] %a, %b : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
}
