// RUN: loom --adg %s

// A valid fabric.pe where all ports are native.
fabric.module @test_native(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %sum = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : !dataflow.bits<32>
}

// A valid fabric.pe where all ports are tagged.
fabric.module @test_tagged(
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
