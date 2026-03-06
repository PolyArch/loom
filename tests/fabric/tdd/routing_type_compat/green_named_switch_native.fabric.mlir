// RUN: loom --adg %s

// Named switch with bit-width-compatible native types: i32 inputs, f32 outputs.
fabric.switch @compat_named_sw : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)

fabric.module @test_named_switch_native(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %o0, %o1 = fabric.switch %a, %b : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %o0 : !dataflow.bits<32>
}
