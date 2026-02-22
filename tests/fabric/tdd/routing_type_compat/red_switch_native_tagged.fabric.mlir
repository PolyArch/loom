// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Switch with category mismatch: i32 input vs tagged<i32,i4> output.
fabric.module @test_switch_native_tagged(
  %a: i32, %b: i32
) -> (!dataflow.tagged<i32, i4>) {
  %o0, %o1 = fabric.switch %a, %b
    : i32 -> !dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>
  fabric.yield %o0 : !dataflow.tagged<i32, i4>
}
