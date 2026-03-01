// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_SW_IMPLICIT_HOLE

// Explicit invalid at slot 1, but implicit hole between slot 2 and slot 4.
fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>, %b: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 5, connectivity_table = [1, 1, 1, 1]]
      {route_table = [
        "route_table[0]: when(tag=0) O[0]<-I[0]",
        "route_table[1]: invalid",
        "route_table[2]: when(tag=1) O[1]<-I[1]",
        "route_table[4]: when(tag=2) O[0]<-I[1]"
      ]}
      %a, %b : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
}
