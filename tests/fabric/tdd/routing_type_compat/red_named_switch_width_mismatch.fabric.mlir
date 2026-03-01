// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: all ports must have bit-width-compatible types

// Named switch with incompatible bit widths: i32 inputs, i16 outputs.
fabric.switch @bad_named_sw : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<16>, !dataflow.bits<16>)

fabric.module @test_named_switch_mismatch(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %o0, %o1 = fabric.switch %a, %a : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o0 : !dataflow.bits<32>
}
