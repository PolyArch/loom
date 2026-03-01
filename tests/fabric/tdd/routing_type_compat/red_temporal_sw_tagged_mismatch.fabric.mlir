// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Temporal switch with tagged value width mismatch:
// tagged<i32,i4> vs tagged<i16,i4>.
fabric.module @test_tsw_tagged_mismatch(
  %a: !dataflow.tagged<!dataflow.bits<32>, i4>,
  %b: !dataflow.tagged<!dataflow.bits<32>, i4>
) -> (!dataflow.tagged<!dataflow.bits<16>, i4>) {
  %o0, %o1 = fabric.temporal_sw [num_route_table = 4]
    %a, %b : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<16>, i4>, !dataflow.tagged<!dataflow.bits<16>, i4>
  fabric.yield %o0 : !dataflow.tagged<!dataflow.bits<16>, i4>
}
