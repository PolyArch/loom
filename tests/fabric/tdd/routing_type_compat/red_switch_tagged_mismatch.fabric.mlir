// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Switch with tagged value width mismatch: tagged<i32,i4> vs tagged<i16,i4>.
fabric.module @test_switch_tagged_mismatch(
  %a: !dataflow.tagged<i32, i4>,
  %b: !dataflow.tagged<i32, i4>
) -> (!dataflow.tagged<i16, i4>) {
  %o0, %o1 = fabric.switch %a, %b
    : !dataflow.tagged<i32, i4> -> !dataflow.tagged<i16, i4>, !dataflow.tagged<i16, i4>
  fabric.yield %o0 : !dataflow.tagged<i16, i4>
}
