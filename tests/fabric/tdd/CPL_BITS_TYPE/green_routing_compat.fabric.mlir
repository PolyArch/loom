// RUN: loom --adg %s

// PE(bits<32>) -> Switch(bits<32>) routing is compatible (same bit width).
fabric.module @test_routing_compat(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %sum = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  %routed = fabric.switch %sum : !dataflow.bits<32> -> !dataflow.bits<32>
  fabric.yield %routed : !dataflow.bits<32>
}
