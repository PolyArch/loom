// RUN: loom --adg %s

// A valid 2x2 switch where every output row has at least one connection.
fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o1, %o2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o1, %o2 : !dataflow.bits<32>, !dataflow.bits<32>
}
