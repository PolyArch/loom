// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with a back-edge through dataflow.carry: %acc reads %next which is
// produced by a textually-later arith.addi. Because the fabric.fu body
// is a graph region (RegionKind::Graph), the textual back-reference is
// well-formed; the enumerator's two-pass materializer must walk firing
// ops and synthesize placeholder operands on first reference, then
// rewire them to the real sw values once the producer is built.
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout.

// CHECK-LABEL: fabric.module @fu_carry_self_feedback
fabric.module @fu_carry_self_feedback {
  %cond = builtin.unrealized_conversion_cast to !fabric.bits<1>
  %init = builtin.unrealized_conversion_cast to !fabric.bits<1>
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

// The materialized subgraph must reproduce the cycle: dataflow.carry
// reads the result of arith.addi which itself reads back the carry.
// CHECK: dataflow.subgraph
// CHECK: %[[ACC:.*]] = dataflow.carry %{{.*}}, %{{.*}}, %[[NEXT:.*]] : i1
// CHECK: %[[NEXT]] = arith.addi %[[ACC]], %{{.*}} : i1
// CHECK: dataflow.yield %[[ACC]] : i1
