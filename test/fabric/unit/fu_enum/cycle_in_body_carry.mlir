// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with a back-edge through dataflow.carry: %acc reads %next which is
// produced by a textually-later arith.addi. Because the fabric.fu body
// is a graph region (RegionKind::Graph), the textual back-reference is
// well-formed; the enumerator's two-pass materializer must walk firing
// ops and synthesize placeholder operands on first reference, then
// rewire them to the real sw values once the producer is built.

// CHECK-LABEL: @fu_carry_self_feedback
func.func @fu_carry_self_feedback(%cond: !fabric.bits<1>,
                                  %init: !fabric.bits<32>) {
  %r = fabric.fu(%c = %cond : !fabric.bits<1>,
                 %i = %init : !fabric.bits<32>) -> !fabric.bits<32> {
    %acc = fabric.op [@dataflow.carry] (%c, %i, %next)
           : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
             -> !fabric.bits<32>
    %next = fabric.op [@arith.addi] (%acc, %i)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %acc : !fabric.bits<32>
  }

  // The materialized subgraph must reproduce the cycle: dataflow.carry
  // reads the result of arith.addi which itself reads back the carry.
  // CHECK: dataflow.subgraph
  // CHECK: %[[ACC:.*]] = dataflow.carry %{{.*}}, %{{.*}}, %[[NEXT:.*]] : i32
  // CHECK: %[[NEXT]] = arith.addi %[[ACC]], %{{.*}} : i32
  // CHECK: dataflow.yield %[[ACC]] : i32

  return
}
