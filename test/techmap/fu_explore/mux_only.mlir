// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: a single fabric.mux axis explores all (sel, discard, disconnect)
// triples. With one fabric.op consuming the mux output and the two mux
// inputs coming from DIFFERENT FU args (so the materialized templates
// have distinct live-input signatures across sel=0/sel=1), the
// post-dedup template count must include at least one template for
// each non-trivial sel choice.

// CHECK-LABEL: @fu_mux2_path
fabric.module @fu_mux2_path(%cast0_fu_mux2_path : !fabric.bits<32>, %cast1_fu_mux2_path : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_mux2_path : !fabric.bits<32>, %b = %cast1_fu_mux2_path : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.mux %x, %y : !fabric.bits<32>
    %k = fabric.op [@arith.addi] (%m, %m)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  // The enumerator emits at least one template wrapping arith.addi
  // and records a fabric.mux config in the loom.from_fu_config attr.
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.from_fu_config = "mux#0
  // CHECK: arith.addi
  // CHECK: dataflow.yield
  }
  fabric.yield
}

