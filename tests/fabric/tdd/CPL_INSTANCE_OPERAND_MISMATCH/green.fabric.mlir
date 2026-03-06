// RUN: loom --adg %s

// A valid fabric.instance whose operand count and types match the target.
fabric.module @alu(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %sum = fabric.pe %a, %b : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : !dataflow.bits<32>
}

fabric.module @top(%p: !dataflow.bits<32>, %q: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @alu(%p, %q) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
