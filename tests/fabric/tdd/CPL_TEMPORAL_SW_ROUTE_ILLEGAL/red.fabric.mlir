// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_SW_ROUTE_ILLEGAL

// Route slot 0 routes output 0 from input 1, but connectivity_table[0][1] = 0 (not connected).
// connectivity_table (2 outputs, 2 inputs): row 0 = [1, 0], row 1 = [0, 1].
fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>, %b: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 2, connectivity_table = [1, 0, 0, 1]]
      {route_table = ["route_table[0]: when(tag=0) O[0]<-I[1]"]}
      %a, %b : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
}
