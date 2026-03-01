// RUN: loom --adg %s

// Switch with bits<32> ports passes verification.
fabric.module @test_switch_bits(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o0, %o1 = fabric.switch %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o0, %o1 : !dataflow.bits<32>, !dataflow.bits<32>
}
