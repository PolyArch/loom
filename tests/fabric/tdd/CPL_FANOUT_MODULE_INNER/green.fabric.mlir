// RUN: loom --adg %s

// Each instance output has at most one consumer, thanks to switch broadcast.
fabric.pe @add(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: !dataflow.bits<32>, %y: !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  %sum = fabric.instance @add(%x, %y) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  // Use switch broadcast to duplicate %sum for two yield operands
  %dup:2 = fabric.switch [connectivity_table = [1, 1]] %sum : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %dup#0, %dup#1 : !dataflow.bits<32>, !dataflow.bits<32>
}
