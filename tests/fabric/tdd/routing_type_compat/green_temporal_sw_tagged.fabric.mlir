// RUN: loom --adg %s

// Temporal switch with bit-width-compatible tagged types:
// tagged<i32,i4> inputs, tagged<f32,i4> outputs.
fabric.module @test_tsw_tagged_compat(
  %a: !dataflow.tagged<i32, i4>,
  %b: !dataflow.tagged<i32, i4>
) -> (!dataflow.tagged<f32, i4>) {
  %o0, %o1 = fabric.temporal_sw [num_route_table = 4]
    %a, %b : !dataflow.tagged<i32, i4> -> !dataflow.tagged<f32, i4>, !dataflow.tagged<f32, i4>
  fabric.yield %o0 : !dataflow.tagged<f32, i4>
}
