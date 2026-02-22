// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Named switch with category mismatch: i32 input vs tagged<i32,i4> output.
fabric.switch @bad_native_tagged_sw : (i32, i32) -> (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>)

fabric.module @test_named_switch_native_tagged(%a: i32) -> (i32) {
  %o0, %o1 = fabric.switch %a, %a : i32 -> i32, i32
  fabric.yield %o0 : i32
}
