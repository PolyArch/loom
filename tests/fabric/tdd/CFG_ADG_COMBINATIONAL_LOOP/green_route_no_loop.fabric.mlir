// RUN: loom --adg %s

// Two switches with diagonal connectivity (out0<-in0, out1<-in1 only) and
// route_table that only enables those diagonal paths. Crossed feedback wires
// cannot form a combinational cycle because the routes do not connect
// in1 -> out0.
fabric.module @top(%x: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  // Use switch broadcast to duplicate %x for two consumers
  %bcast:2 = fabric.switch [connectivity_table = [1, 1]] {route_table = [1, 1]} %x : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %sw0:2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] {route_table = [1, 1]} %bcast#0, %sw1#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %sw1:2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] {route_table = [1, 1]} %bcast#1, %sw0#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %sw0#1 : !dataflow.bits<32>
}
