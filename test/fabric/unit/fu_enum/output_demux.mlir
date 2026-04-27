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

  // sel=0 and sel=1 each produce a 1-result subgraph wrapping just the
  // arith.muli (the demux is purely a routing detail). They are
  // graph-isomorphic and dedup keeps only the lex-smallest.
  // CHECK: dataflow.subgraph
  // CHECK-SAME: demux#0{sel=0,discard=false,disconnect=false}
  // CHECK:   %[[M0:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
  // CHECK:   dataflow.yield %[[M0]] : i32

  // No second template emitted.
  // CHECK-NOT: demux#0{sel=1,discard=false,disconnect=false}

  // Discard / disconnect configs leave both yields dead so no candidate
  // should be emitted for them.
  // CHECK-NOT: discard=true
  // CHECK-NOT: disconnect=true

  return
}
