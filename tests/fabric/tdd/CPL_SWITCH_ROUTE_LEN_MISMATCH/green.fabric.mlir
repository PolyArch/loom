// RUN: loom --adg %s

// A valid 2x2 switch with route_table length matching popcount of
// connectivity_table (2 ones -> route_table length 2).
fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] {route_table = [1, 0]} %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o1, %o2 : !dataflow.bits<32>, !dataflow.bits<32>
}
