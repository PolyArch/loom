// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Switch with bit-width mismatch: bits<32> input vs bits<16> output.
// (Previously tested tagged value width mismatch, but regular switches
// no longer accept tagged types.)
fabric.module @test_switch_tagged_mismatch(
  %a: !dataflow.bits<32>,
  %b: !dataflow.bits<32>
) -> (!dataflow.bits<16>) {
  %o0, %o1 = fabric.switch %a, %b
    : !dataflow.bits<32> -> !dataflow.bits<16>, !dataflow.bits<16>
  fabric.yield %o0 : !dataflow.bits<16>
}
