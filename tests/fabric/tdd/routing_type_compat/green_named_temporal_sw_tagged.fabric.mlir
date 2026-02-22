// RUN: loom --adg %s

// Named temporal_sw with bit-width-compatible tagged types:
// tagged<i32,i4> inputs, tagged<f32,i4> outputs.
fabric.temporal_sw @compat_named_tsw [num_route_table = 4]
  : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>) -> (!dataflow.tagged<f32, i4>, !dataflow.tagged<f32, i4>)

fabric.module @test_named_tsw_tagged(
  %a: !dataflow.tagged<i32, i4>,
  %b: !dataflow.tagged<i32, i4>
) -> (!dataflow.tagged<i32, i4>) {
  %o0, %o1 = fabric.temporal_sw [num_route_table = 4]
    %a, %b : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
  fabric.yield %o0 : !dataflow.tagged<i32, i4>
}
