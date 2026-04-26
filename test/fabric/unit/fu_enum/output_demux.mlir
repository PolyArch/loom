// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with an arith.muli feeding a fabric.demux at the output stage. Both
// demux outputs are FU yields, but a fabric.demux is a 1-of-N selector:
// only outputs[sel] carries a value in any given configuration. The
// enumerator must therefore emit per-config single-result subgraph
// templates (one for sel=0, one for sel=1), each wrapping just the muli.
// Discard / disconnect configs leave both yield positions inactive and
// must produce no candidate at all.

// CHECK-LABEL: @fu_muli_output_demux
func.func @fu_muli_output_demux(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r0, %r1 = fabric.fu(%x = %a : !fabric.bits<32>,
                       %y = %b : !fabric.bits<32>)
                       -> (!fabric.bits<32>, !fabric.bits<32>) {
    %p = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %d0, %d1 = fabric.demux %p : !fabric.bits<32> -> 2
    fabric.yield %d0, %d1 : !fabric.bits<32>, !fabric.bits<32>
  }

  // sel=0 candidate: only %d0 is live; subgraph signature shrinks to a
  // single i32 result.
  // CHECK: dataflow.subgraph
  // CHECK-SAME: demux#0{sel=0,discard=false,disconnect=false}
  // CHECK:   %[[M0:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
  // CHECK:   dataflow.yield %[[M0]] : i32

  // sel=1 candidate: only %d1 is live; same single-result shape.
  // CHECK: dataflow.subgraph
  // CHECK-SAME: demux#0{sel=1,discard=false,disconnect=false}
  // CHECK:   %[[M1:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
  // CHECK:   dataflow.yield %[[M1]] : i32

  // Discard / disconnect configs leave both yields dead so no candidate
  // should be emitted for them.
  // CHECK-NOT: discard=true
  // CHECK-NOT: disconnect=true

  return
}
