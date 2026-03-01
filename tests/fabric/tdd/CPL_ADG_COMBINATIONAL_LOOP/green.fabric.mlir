// RUN: loom --adg %s

// Combinational cycle broken by fabric.fifo (sequential element).
fabric.module @test_comb_loop_ok(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %f = fabric.fifo [depth = 2] %sw1#0 : !dataflow.bits<32>
  %sw0:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %f : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %sw1:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %sw0#0, %sw0#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %sw1#1 : !dataflow.bits<32>
}
