// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Switch with bit-width mismatch: bits<32> inputs vs bits<64> outputs.
// (Previously tested native vs tagged category mismatch, but regular switches
// no longer accept tagged types.)
fabric.module @test_switch_native_tagged(
  %a: !dataflow.bits<32>, %b: !dataflow.bits<32>
) -> (!dataflow.bits<64>) {
  %o0, %o1 = fabric.switch %a, %b
    : !dataflow.bits<32> -> !dataflow.bits<64>, !dataflow.bits<64>
  fabric.yield %o0 : !dataflow.bits<64>
}
