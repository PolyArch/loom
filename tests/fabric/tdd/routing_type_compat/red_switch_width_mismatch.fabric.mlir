// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Switch with incompatible bit widths: i32 inputs, i16 outputs.
fabric.module @test_switch_width_mismatch(%a: i32, %b: i32) -> (i16) {
  %o0, %o1 = fabric.switch %a, %b : i32 -> i16, i16
  fabric.yield %o0 : i16
}
