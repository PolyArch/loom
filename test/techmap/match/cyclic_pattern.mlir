// RUN: loom %s -loom-map-subgraph-to-fus 2>&1 | FileCheck %s

// Pins: a cyclic user pattern (carry + addi self-feedback) should match
// an FU whose body holds the same cyclic structure. The fabric.fu body
// is a graph region, so a textual back-reference (`%next`) is
// well-formed and the enumerator handles the cycle via its two-pass
// materializer.
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout (carry's TypeParam(0) data ports accept any width); the
// pattern is correspondingly typed as i1.

// CHECK: loom.matched_fu
fabric.module @hw_carry_loop(%cond : !fabric.bits<1>, %init : !fabric.bits<1>) {
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

func.func @pat_carry_loop(%cond: i1, %init: i1) -> i1 {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i1) -> i1
       attributes {loom.is_pattern} {
    %acc = dataflow.carry %c, %i, %next : i1
    %next = arith.addi %acc, %i : i1
    dataflow.yield %acc : i1
  }
  return %r : i1
}
