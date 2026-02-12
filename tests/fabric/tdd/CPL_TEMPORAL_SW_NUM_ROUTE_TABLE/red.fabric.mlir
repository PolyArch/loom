// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_SW_NUM_ROUTE_TABLE

// num_route_table = 0 is invalid (must be >= 1).
fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 0, connectivity_table = [1, 1, 1, 1]] %a, %b : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
}
