// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Named switch with incompatible bit widths: i32 inputs, i16 outputs.
fabric.switch @bad_named_sw : (i32, i32) -> (i16, i16)

fabric.module @test_named_switch_mismatch(%a: i32) -> (i32) {
  %o0, %o1 = fabric.switch %a, %a : i32 -> i32, i32
  fabric.yield %o0 : i32
}
