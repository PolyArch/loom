// RUN: loom --adg %s

// Switch with bit-width-compatible tagged types: tagged<i32,i4> and tagged<f32,i4>.
fabric.module @test_switch_tagged_compat(
  %a: !dataflow.tagged<i32, i4>,
  %b: !dataflow.tagged<i32, i4>
) -> (!dataflow.tagged<f32, i4>) {
  %o0, %o1 = fabric.switch %a, %b
    : !dataflow.tagged<i32, i4> -> !dataflow.tagged<f32, i4>, !dataflow.tagged<f32, i4>
  fabric.yield %o0 : !dataflow.tagged<f32, i4>
}
