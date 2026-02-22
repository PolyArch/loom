// RUN: loom --adg %s

// Named switch with bit-width-compatible native types: i32 inputs, f32 outputs.
fabric.switch @compat_named_sw : (i32, i32) -> (f32, f32)

fabric.module @test_named_switch_native(%a: i32, %b: i32) -> (i32) {
  %o0, %o1 = fabric.switch %a, %b : i32 -> i32, i32
  fabric.yield %o0 : i32
}
