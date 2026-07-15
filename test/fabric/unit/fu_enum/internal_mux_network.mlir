// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// fabric.mux placed in the middle of a multi-stage compute network. This
// example exercises real SSA broadcast: %y feeds both stages, so a valid
// configuration must keep both consumers active.

// CHECK-LABEL: fabric.module @fu_internal_chain
fabric.module @fu_internal_chain(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>, %d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>,
                    %pd = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%w = %pa : !fabric.bits<32>,
              %x = %pb : !fabric.bits<32>,
              %y = %pc : !fabric.bits<32>,
              %z = %pd : !fabric.bits<32>) -> !fabric.bits<32> {
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
  }
  fabric.yield
}

// m1.sel=0 with m2.sel=0 yields the full muladd compute. m1.sel=1 with
// m2.sel=0 produces the same shape (block-arg permutation only) and is
// deduped to the lex-smallest config.
// CHECK: mux#0{sel=0,discard=false,disconnect=false}; mux#1{sel=0,discard=false,disconnect=false}
// CHECK-NOT: mux#0{sel=1,discard=false,disconnect=false}; mux#1{sel=0,discard=false,disconnect=false}

// m2.sel=1 would leave the multiplier inactive while %y still feeds the
// adder. That configuration requires an implicit drain on the multiplier
// input and must not materialize.
// CHECK-NOT: mux#1{sel=1,discard=false,disconnect=false}
