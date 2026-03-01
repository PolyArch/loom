// RUN: loom --adg %s

// A valid fabric.instance whose result count and types match the target.
fabric.module @inc(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a : (!dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %s = arith.addi %x, %c1 : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}

fabric.module @top(%v: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @inc(%v) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
