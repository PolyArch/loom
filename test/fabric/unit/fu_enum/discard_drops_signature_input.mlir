// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins the invariant that a fabric.mux operating in discard mode does
// NOT contribute its selected input to the materialized subgraph
// signature. Discard mode is a hardware-only drain: the selected input
// is consumed at the FU boundary but never propagates into the software
// graph. Including such block-args in the signature would pad the
// per-config function with parameters that the body never reads, and
// would spuriously distinguish otherwise software-isomorphic templates.
//
// Shape: a fabric.mux feeds one input of a variadic dataflow.sync M=2.
// Sync's bitmask allow-set is {"11","10","01"}: with bitmask=01 only the
// non-mux input drives the surviving output, so the discard-mode mux
// configuration's body should reduce to a one-arg subgraph that reads
// only %z. Pre-fix, the discard-mode template instead exposed both %x
// (or %y, depending on sel) and %z in its signature. After the fix the
// discard-mode N=1 template becomes isomorphic to the bitmask=10
// non-discard one-arg template and dedup folds them together.

// CHECK-LABEL: fabric.module @fu_mux_then_sync
fabric.module @fu_mux_then_sync(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>) {
      %m = fabric.mux %x, %y : !fabric.bits<32>
      %s:2 = fabric.op [@dataflow.sync] (%m, %z)
             {hw_params = [{bitmask = ["11", "10", "01"]}]}
             : (!fabric.bits<32>, !fabric.bits<32>)
               -> (!fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %s#0, %s#1 : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

// First template: full bitmask=11 with normal-mode mux. Reads two
// software inputs (the chosen mux source and %z) and produces both
// sync outputs.
// CHECK: func.func private @fu0_subgraph_0(%arg0: i32, %arg1: i32) -> (i32, i32)
// CHECK: dataflow.subgraph
// CHECK-SAME: bitmask=11
// CHECK: dataflow.sync

// Second template: bitmask=10 with normal-mode mux. Only the mux side
// drives the surviving output, so the signature is one-in one-out.
// CHECK: func.func private @fu0_subgraph_1(%arg0: i32) -> i32
// CHECK: dataflow.subgraph
// CHECK-SAME: bitmask=10

// The discard-mode pairing with bitmask=01 must NOT introduce a third
// distinct template: after the discard-mode signature shrink its
// signature is also (i32) -> i32 reading only %z, so it is
// software-isomorphic to subgraph_1 and the dedup pass collapses it.
// CHECK-NOT: func.func private @fu0_subgraph_2
