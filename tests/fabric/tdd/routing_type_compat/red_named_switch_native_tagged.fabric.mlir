// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Named switch with bit-width mismatch: bits<32> input vs bits<16> output.
// (Previously tested native vs tagged category mismatch, but regular switches
// no longer accept tagged types.)
fabric.switch @bad_width_mismatch_sw : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<16>, !dataflow.bits<16>)

fabric.module @test_named_switch_native_tagged(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %o0, %o1 = fabric.switch %a, %a : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o0 : !dataflow.bits<32>
}
