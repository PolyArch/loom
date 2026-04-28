// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins the producer-fanout-into-distinct-muxes shape: a fabric.op
// (%pre = addi(D, B)) feeds two structurally-different fabric.mux
// instances whose outputs converge into a single multi-arg fabric.op
// (%m = muli(%xa, %xb)). The two muxes pick distinct first-input
// block-args (%A vs %B) so they are NOT structurally identical, which
// is the exact enumerator failure mode that motivates this test.
//
// Correct alive analysis must recognize that a firing fabric.mux drains
// every input port (the selected port propagates, the non-selected
// ports complete their handshakes by accepting and discarding the
// data). Without that recognition the enumerator killed %pre whenever
// even one downstream mux did not select it, collapsing the entire
// candidate set to zero. With the fix, every (mux#0.sel, mux#1.sel)
// combination yields a structurally distinct effective compute, so the
// dedup pass keeps four templates: muli(A, B), muli(pre, B),
// muli(A, pre), muli(pre, pre).
//
// Note: the FU body intentionally does NOT use back-edges; this test
// is independent of graph-region work and exercises the alive
// fixed-point alone.

// CHECK-LABEL: @repro_fanout_converging_muxes
func.func @repro_fanout_converging_muxes(%a: !fabric.bits<32>,
                                          %b: !fabric.bits<32>,
                                          %d: !fabric.bits<32>) {
  %r = fabric.fu(%A = %a : !fabric.bits<32>,
                 %B = %b : !fabric.bits<32>,
                 %D = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    %pre = fabric.op [@arith.addi] (%D, %B)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %xa = fabric.mux %A, %pre : !fabric.bits<32>
    %xb = fabric.mux %B, %pre : !fabric.bits<32>
    %m = fabric.op [@arith.muli] (%xa, %xb)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %m : !fabric.bits<32>
  }

  // Sanity floor: at least four distinct private subgraph wrappers must
  // be emitted, one per (mux#0.sel, mux#1.sel) combination.
  // CHECK: func.func private @fu0_subgraph_0
  // CHECK: func.func private @fu0_subgraph_1
  // CHECK: func.func private @fu0_subgraph_2
  // CHECK: func.func private @fu0_subgraph_3

  // At least one of the emitted templates must contain the converging
  // arith.muli, confirming the downstream multi-arg op materializes.
  // CHECK: arith.muli

  return
}
