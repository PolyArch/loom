// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: a multi-op template whose body contains a back-edge (carry's
// carry-input fed by a later arith.addi) is a legal graph-region template
// under the dataflow.subgraph contract. The partitioner must accept such
// a template and fuse the matching cyclic chain in the user graph into a
// single dataflow.subgraph -- not split it across two single-op blocks.
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout (carry's TypeParam(0) data ports accept any width); the
// pattern is correspondingly typed as i1.

// CHECK-LABEL: @fu_carry_loop
fabric.module @fu_carry_loop(%cond : !fabric.bits<1>, %init : !fabric.bits<1>) {
  fabric.spatial_pe(%pcond = %cond : !fabric.bits<1>,
                    %pinit = %init : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%c = %pcond : !fabric.bits<1>,
              %i = %pinit : !fabric.bits<1>) -> !fabric.bits<1> {
      %acc = fabric.op [@dataflow.carry] (%c, %i, %next)
             : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
               -> !fabric.bits<1>
      %next = fabric.op [@arith.addi] (%acc, %i)
              : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %acc : !fabric.bits<1>
    }
  }
  fabric.yield
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
func.func @graph_carry_loop(%cond: i1, %init: i1) -> i1 {
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i1) -> i1 {
    %acc = dataflow.carry %c, %i, %next : i1
    %next = arith.addi %acc, %i : i1
    dataflow.yield %acc : i1
  }
  return %r : i1
}
