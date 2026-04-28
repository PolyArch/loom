// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: a graph-region cycle closed by dataflow.carry must NOT crash the
// partitioner and must NOT introduce a multi-block SSA cycle. The
// canonical safe shape (greedy default): the carry is wrapped in its
// own dataflow.subgraph; the loop-closing arith.addi stays at graph
// level so the cycle is contained inside the graph block. The
// no-multi-block-cycle invariant: no two dataflow.subgraph blocks
// reference each other.

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

// CHECK-LABEL: @fu_carry
func.func @fu_carry(%cond: !fabric.bits<1>,
                    %init: !fabric.bits<32>,
                    %carry: !fabric.bits<32>) {
  %r = fabric.fu(%c = %cond : !fabric.bits<1>,
                 %i = %init : !fabric.bits<32>,
                 %k = %carry : !fabric.bits<32>) -> !fabric.bits<32> {
    %o = fabric.op [@dataflow.carry] (%c, %i, %k)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %o : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_self_feedback
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: dataflow.carry
// CHECK-NEXT: dataflow.yield
// CHECK: arith.addi
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_self_feedback(%cond: i1, %init: i32, %step: i32) -> i32 {
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32, %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}
