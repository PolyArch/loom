// RUN: loom --adg %s

// Inline switch cycle broken by inline fifo (sequential element).
// Switches have route_table enabling all paths, but fifo breaks the cycle.
fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %f = fabric.fifo [depth = 2] %sw1#0 : !dataflow.bits<32>
  %sw0:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] {route_table = [1, 1, 1, 1]} %a, %f : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %sw1:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] {route_table = [1, 1, 1, 1]} %sw0#0, %sw0#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %sw1#1 : !dataflow.bits<32>
}
