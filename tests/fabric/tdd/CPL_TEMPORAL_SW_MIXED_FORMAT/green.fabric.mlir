// RUN: loom --adg %s

// Valid: all route_table entries use human-readable format.
fabric.module @test(%a: !dataflow.tagged<i32, i4>, %b: !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 2, connectivity_table = [1, 1, 1, 1]]
      {route_table = [
        "route_table[0]: when(tag=0) O[0]<-I[0]",
        "route_table[1]: when(tag=1) O[1]<-I[1]"
      ]}
      %a, %b : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
}
