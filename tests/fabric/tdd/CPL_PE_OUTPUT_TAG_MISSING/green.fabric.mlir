// RUN: loom --adg %s

// A valid tagged fabric.pe with the required output_tag attribute.
fabric.module @test(
    %a: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %b: !dataflow.tagged<!dataflow.bits<32>, i4>
) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %sum = fabric.pe %a, %b
      {output_tag = [0 : i4]}
      : (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>)
      -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : !dataflow.tagged<!dataflow.bits<32>, i4>
}
