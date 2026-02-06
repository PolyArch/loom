// RUN: loom --adg %s

// A valid temporal_sw where all route_table entries reference connected positions.
// connectivity_table (2 outputs, 2 inputs): row 0 = [1, 0], row 1 = [0, 1].
// Route slot 0 routes output 0 from input 0 (connected), slot 1 routes output 1 from input 1 (connected).
fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 2, connectivity_table = [1, 0, 0, 1]] {route_table = [[[0 : i64, 0 : i64]], [[1 : i64, 1 : i64]]]} %a, %b : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
}
