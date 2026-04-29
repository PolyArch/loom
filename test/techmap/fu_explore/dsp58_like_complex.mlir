// RUN: loom %s -loom-enumerate-fu-subgraphs 2>&1 | FileCheck %s

// Models a simplified AMD/Xilinx DSP58-style fabric.fu so we can inspect
// the full set of materialized software graphs. The native DSP58 carries
// 27x24 multiplier widths and a 58-bit accumulator; here every datapath
// is widened/narrowed uniformly to bits<1> so the focus stays on the
// configuration combinatorics rather than width arithmetic and the FU
// fits the spatial_pe uniform-W rule.
//
// Datapath (mirroring the public DSP58 simplified block diagram):
//
//                       D --+
//                           | (pre-adder share group: addi/subi)
//                       B --+ --[+/-]--> %pre
//                                              \
//                          A ---------(mux_a)---+--> %xa --+
//                          B ---------(mux_b)---+--> %xb --+--[muli]--> %m
//
//                       %m -----(acc_op share group: addi/subi)-----.
//                                                                   |
//                                          (back-edge from %next)<--+
//                                                                   |
//                                       dataflow.carry (cond=NEGATE)|
//                                       reads %next; emits %acc <---'
//
//                       %acc ---[xori with C]---> XOR output
//                       %acc ---[cmpi vs  C]----> Pattern Detect
//                       %acc -------------------> P output
//
// Configurable axes (sw_configs explored by the enumerator):
//   * pre-adder op_sel  -> 2 alternatives   (addi / subi)
//   * mux_a (2-input)   -> 5 alternatives   (sel=0|1, discard*2, disconnect)
//   * mux_b (2-input)   -> 5 alternatives   (sel=0|1, discard*2, disconnect)
//   * acc op_sel        -> 2 alternatives   (addi / subi)
//   * cmpi predicate    -> 4 alternatives   (eq, ne, slt, sgt) via hw_params
//
// Cartesian envelope: 2 * 5 * 5 * 2 * 4 = 400 raw configurations.
// Most disconnect/discard configurations leave the multiplier or the
// accumulator with a missing operand and prune to no template; the
// enumerator's effective-config dedup pass collapses isomorphic
// survivors. The final emitted set is intended to be browsed manually:
//
//   loom %s -loom-enumerate-fu-subgraphs
//
// Pins below assert that the enumerator emits a non-trivial number of
// distinct templates and that a few signature shapes are present.

// CHECK-LABEL: fabric.module @dsp58_like
fabric.module @dsp58_like(%n : !fabric.bits<1>, %a : !fabric.bits<1>, %b : !fabric.bits<1>, %c : !fabric.bits<1>, %d : !fabric.bits<1>) {
  fabric.spatial_pe(%pn = %n : !fabric.bits<1>,
                    %pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>,
                    %pc = %c : !fabric.bits<1>,
                    %pd = %d : !fabric.bits<1>)
                   -> (!fabric.bits<1>,
                 !fabric.bits<1>,
                 !fabric.bits<1>) {
    fabric.fu(%negate = %pn : !fabric.bits<1>,
              %A = %pa : !fabric.bits<1>,
              %B = %pb : !fabric.bits<1>,
              %C = %pc : !fabric.bits<1>,
              %D = %pd : !fabric.bits<1>)
             -> (!fabric.bits<1>,
                 !fabric.bits<1>,
                 !fabric.bits<1>) {
      // Pre-adder: D op B with op selectable from {addi, subi}.
      %pre = fabric.op [@arith.addi, @arith.subi] (%D, %B)
             : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>

      // Multiplier-side input muxes: each is a 2-input fabric.mux.
      %xa = fabric.mux %A, %pre : !fabric.bits<1>
      %xb = fabric.mux %pre, %B : !fabric.bits<1>

      // Multiplier (widths uniform; native DSP58 is 27x24).
      %m = fabric.op [@arith.muli] (%xa, %xb)
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>

      // Accumulator stage with back-edge: %acc reads %next which is
      // produced by a textually-later arith.{addi,subi}. fabric.fu's
      // graph-region semantics permit this forward reference; the
      // enumerator's two-pass materializer threads the back-edge into
      // each emitted dataflow.subgraph.
      %acc = fabric.op [@dataflow.carry] (%negate, %A, %next)
             : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
               -> !fabric.bits<1>
      %next = fabric.op [@arith.addi, @arith.subi] (%acc, %m)
              : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>

      // XOR output: bit-wise xori of accumulator with C.
      %xor = fabric.op [@arith.xori] (%acc, %C)
             : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>

      // Pattern detector: cmpi(%acc, %C) with the predicate restricted by
      // hw_params to {eq, ne, slt, sgt} (the enumerator treats this as a
      // 4-way axis; without hw_params it would be the full 10-way set).
      %pat = fabric.op [@arith.cmpi] (%acc, %C)
             {hw_params = [{predicate = ["eq", "ne", "slt", "sgt"]}]}
             : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>

      fabric.yield %acc, %xor, %pat
          : !fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>
    }
  }
  fabric.yield
}

// At least 8 distinct templates must be emitted (sanity floor; on this
// FU the actual count is in the dozens). Keeping the lower bound loose
// so future dedup tightening does not break the pin.
// CHECK: func.func private @fu0_subgraph_0
// CHECK: func.func private @fu0_subgraph_1
// CHECK: func.func private @fu0_subgraph_2
// CHECK: func.func private @fu0_subgraph_3
// CHECK: func.func private @fu0_subgraph_4
// CHECK: func.func private @fu0_subgraph_5
// CHECK: func.func private @fu0_subgraph_6
// CHECK: func.func private @fu0_subgraph_7

// Some emitted template must contain the back-edge through dataflow.carry,
// confirming graph-region materialization works end-to-end.
// CHECK: dataflow.carry
