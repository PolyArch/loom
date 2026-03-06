// RUN: loom --adg %s

// A valid temporal_sw with port counts within the 32-port limit.
fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>, %b: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>) {
  %o1, %o2 = fabric.temporal_sw [num_route_table = 2, connectivity_table = [1, 1, 1, 1]] %a, %b : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %o1, %o2 : !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>
}
