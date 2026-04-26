// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Same chain as greedy_two_op_chain.mlir but with two single-op FUs only:
// one for arith.muli and one for arith.addi. There is no template that
// spans the chain, so greedy must emit two singleton dataflow.subgraphs.

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

// CHECK-LABEL: @graph_two_op
// First subgraph wraps the arith.muli.
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: dataflow.yield
// Second subgraph wraps the arith.addi.
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
func.func @graph_two_op(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
