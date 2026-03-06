// RUN: loom --adg %s

// Switch with bit-width-compatible native types: i32 inputs, f32 outputs.
fabric.module @test_switch_native_compat(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %o0, %o1 = fabric.switch %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o0 : !dataflow.bits<32>
}
