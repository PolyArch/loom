// RUN: loom --adg %s

// Module with bits<32> ports (correct).
fabric.pe @add(%p0: !dataflow.bits<32>, %p1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @good_bits_port(%x: !dataflow.bits<32>, %y: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.instance @add(%x, %y) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %r : !dataflow.bits<32>
}
