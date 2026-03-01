// RUN: loom --adg %s

// Named switch definition with bits<32> ports.
fabric.switch @sw_bits : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)

fabric.module @test_named_switch_bits(
    %a: !dataflow.bits<32>, %b: !dataflow.bits<32>
) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %o0, %o1 = fabric.instance @sw_bits(%a, %b)
      : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  fabric.yield %o0, %o1 : !dataflow.bits<32>, !dataflow.bits<32>
}
