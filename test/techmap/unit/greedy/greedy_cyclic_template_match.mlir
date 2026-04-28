// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: a multi-op template whose body contains a back-edge (carry's
// carry-input fed by a later arith.addi) is a legal graph-region template
// under the dataflow.subgraph contract. The partitioner must accept such
// a template and fuse the matching cyclic chain in the user graph into a
// single dataflow.subgraph -- not split it across two single-op blocks.

// CHECK-LABEL: @fu_carry_loop
func.func @fu_carry_loop(%cond: !fabric.bits<1>, %init: !fabric.bits<32>) {
  %r = fabric.fu(%c = %cond : !fabric.bits<1>,
                 %i = %init : !fabric.bits<32>) -> !fabric.bits<32> {
    %acc = fabric.op [@dataflow.carry] (%c, %i, %next)
           : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
             -> !fabric.bits<32>
    %next = fabric.op [@arith.addi] (%acc, %i)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %acc : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_carry_loop
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK: dataflow.carry
// CHECK: arith.addi
// CHECK: dataflow.yield
// Only one subgraph should be emitted -- both body ops fuse into it.
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_carry_loop(%cond: i1, %init: i32) -> i32 {
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %i : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}
