// RUN: loom --adg %s

// Switch with bit-width-compatible native types: i32 inputs, f32 outputs.
fabric.module @test_switch_native_compat(%a: i32, %b: i32) -> (f32) {
  %o0, %o1 = fabric.switch %a, %b : i32 -> f32, f32
  fabric.yield %o0 : f32
}
