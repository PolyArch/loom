// RUN: loom --adg %s

// Switch with matching bit-width-compatible types: all ports are bits<32>.
// (Previously tested tagged types on switches, but regular switches now
// only accept BitsType/NoneType; tagged routing uses temporal_sw.)
fabric.module @test_switch_tagged_compat(
  %a: !dataflow.bits<32>,
  %b: !dataflow.bits<32>
) -> (!dataflow.bits<32>) {
  %o0, %o1 = fabric.switch %a, %b
    : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o0 : !dataflow.bits<32>
}
