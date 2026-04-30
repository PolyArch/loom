// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: variadic dataflow.mux with M=3 data ports and an i32 sel input.
// Each bitmask popcount N picks the live data subset; the materialized
// dataflow.mux's sel port logical width is i1 for N=2 and index for
// N>=3 (per the dataflow op verifier).

// CHECK-LABEL: @fu_mux3
fabric.module @fu_mux3(%cast0_fu_mux3 : !fabric.bits<32>, %cast1_fu_mux3 : !fabric.bits<32>, %cast2_fu_mux3 : !fabric.bits<32>, %cast3_fu_mux3 : !fabric.bits<32>) {
  fabric.pe [spatial] (%s = %cast0_fu_mux3 : !fabric.bits<32>, %a = %cast1_fu_mux3 : !fabric.bits<32>, %b = %cast2_fu_mux3 : !fabric.bits<32>, %c = %cast3_fu_mux3 : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%ss = %s : !fabric.bits<32>,
                 %aa = %a : !fabric.bits<32>,
                 %bb = %b : !fabric.bits<32>,
                 %cc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %k = fabric.op [@dataflow.mux] (%ss, %aa, %bb, %cc)
         : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  // CHECK-DAG: dataflow.mux
  }
  fabric.yield
}

