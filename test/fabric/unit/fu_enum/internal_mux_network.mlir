// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// fabric.mux placed in the middle of a multi-stage compute network. This
// example also exercises SSA fan-out (%y feeds both stages) and the strict
// deadlock invariant: only configurations where both stages fire are
// valid, since %y can only broadcast when every consumer port is active.

// CHECK-LABEL: @fu_internal_chain
func.func @fu_internal_chain(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                              %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  %r = fabric.fu(%w = %a : !fabric.bits<32>,
                 %x = %b : !fabric.bits<32>,
                 %y = %c : !fabric.bits<32>,
                 %z = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    // Stage 1: mux selects one of (w, x) into a multiplier; %y feeds both
    // the multiplier and the downstream adder.
    %m1 = fabric.mux %w, %x : !fabric.bits<32>
    %p = fabric.op [@arith.muli] (%m1, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    // Stage 2: mux selects one of (p, z) into the final adder.
    %m2 = fabric.mux %p, %z : !fabric.bits<32>
    %s = fabric.op [@arith.addi] (%m2, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %s : !fabric.bits<32>
  }

  // m1.sel=0 and m1.sel=1 (with m2.sel=0) yield graph-isomorphic
  // subgraphs: each is muladd(blockarg_w_or_x, blockarg_y, blockarg_y).
  // Dedup keeps the lex-smallest (m1.sel=0).
  // CHECK: mux#0{sel=0,discard=false,disconnect=false}; mux#1{sel=0,discard=false,disconnect=false}
  // CHECK-NOT: mux#0{sel=1,discard=false,disconnect=false}; mux#1{sel=0,discard=false,disconnect=false}

  // m2.sel=1 leaves the multiplier silent so %y has an inactive consumer
  // on its multiplier port: the broadcast cannot complete and the config
  // must be dropped.
  // CHECK-NOT: mux#1{sel=1,discard=false,disconnect=false}

  return
}
