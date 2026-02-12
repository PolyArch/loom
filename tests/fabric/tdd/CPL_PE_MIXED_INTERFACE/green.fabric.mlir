// RUN: loom --adg %s

// A valid fabric.pe where all ports are native.
fabric.module @test_native(%a: i32, %b: i32) -> (i32) {
  %sum = fabric.pe %a, %b : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}

// A valid fabric.pe where all ports are tagged.
fabric.module @test_tagged(
    %a: !dataflow.tagged<i32, i4>,
    %b: !dataflow.tagged<i32, i4>
) -> (!dataflow.tagged<i32, i4>) {
  %sum = fabric.pe %a, %b
      {output_tag = [0 : i4]}
      : (!dataflow.tagged<i32, i4>, !dataflow.tagged<i32, i4>)
      -> (!dataflow.tagged<i32, i4>) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : !dataflow.tagged<i32, i4>
}
