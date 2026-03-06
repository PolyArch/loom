// RUN: loom --adg %s

// A valid fabric.pe (inline) with a fabric.instance plus another operation.
// The body has more than just a single fabric.instance, so this is allowed.
fabric.pe @inner_add(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>)
    [latency = [1 : i16, 1 : i16, 1 : i16]]
    -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @test(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %t = fabric.instance @inner_add(%x, %y) : (i32, i32) -> (i32)
    %s = arith.addi %t, %x : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}
