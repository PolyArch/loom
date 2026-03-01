// RUN: loom --adg %s

// Module input is duplicated via switch broadcast before feeding
// two consumers. Each connection is strictly 1-to-1.
fabric.pe @add(%a: !dataflow.bits<32>, %b: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}

fabric.module @top(%x: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  // Use switch broadcast to duplicate %x for two consumers
  %dup:2 = fabric.switch [connectivity_table = [1, 1]] %x : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %sum = fabric.instance @add(%dup#0, %dup#1) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>)
  fabric.yield %sum : !dataflow.bits<32>
}
