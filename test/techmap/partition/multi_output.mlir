// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: a user graph that produces two distinct yields from two ops in
// program order. With per-op singleton FUs available, each op is wrapped
// in its own dataflow.subgraph; the graph yields both subgraph results.

// CHECK-LABEL: @fu_addi
func.func @fu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @fu_muli
func.func @fu_muli(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_two_outputs
// CHECK: dataflow.graph
// CHECK-DAG: arith.addi
// CHECK-DAG: arith.muli
// CHECK: dataflow.yield
func.func @graph_two_outputs(%a: i32, %b: i32) -> (i32, i32) {
  %s, %p = dataflow.graph(%x = %a : i32, %y = %b : i32) -> (i32, i32) {
    %sum = arith.addi %x, %y : i32
    %prod = arith.muli %x, %y : i32
    dataflow.yield %sum, %prod : i32, i32
  }
  return %s, %p : i32, i32
}
