// RUN: loom --adg %s

// A valid acyclic instance chain: @top -> @middle -> @leaf (no cycle).
fabric.module @leaf(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %r = fabric.pe %a : (!dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %s = arith.addi %x, %c1 : i32
    fabric.yield %s : i32
  }
  fabric.yield %r : !dataflow.bits<32>
}

fabric.module @middle(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @leaf(%a) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}

fabric.module @top(%v: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %out = fabric.instance @middle(%v) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %out : !dataflow.bits<32>
}
