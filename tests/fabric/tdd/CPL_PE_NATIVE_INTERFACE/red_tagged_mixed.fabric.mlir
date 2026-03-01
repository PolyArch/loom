// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_NATIVE_INTERFACE

// PE with mixed tagged interface: one port uses tagged<bits<32>,i4>,
// another uses tagged<i32,i4> (native value in tagged is not allowed).
fabric.pe @bad_mixed_tagged(
    %p0: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %p1: !dataflow.tagged<i32, i4>
) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}
