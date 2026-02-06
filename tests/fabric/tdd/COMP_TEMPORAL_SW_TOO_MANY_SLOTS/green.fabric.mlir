// RUN: loom --adg %s

// A valid temporal_sw with route_table slot count (2) equal to num_route_table (2).
fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 2, connectivity_table = [1, 1, 1, 1]] {route_table = [[[0 : i64, 0 : i64]], [[1 : i64, 1 : i64]]]} %a, %b : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
}
