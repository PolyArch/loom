// RUN: loom --adg %s

// A valid fabric.pe body that instances a named fabric.pe (legal target).
fabric.pe @adder(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>)
    [latency = [1 : i16, 1 : i16, 1 : i16]]
    -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
  %s = arith.addi %a, %b : i32
  fabric.yield %s : i32
}

fabric.module @top(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>, %c: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a, %b, %c : (!dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32, %z: i32):
    %out = fabric.instance @adder(%x, %y) : (i32, i32) -> (i32)
    %s = arith.addi %out, %z : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}
