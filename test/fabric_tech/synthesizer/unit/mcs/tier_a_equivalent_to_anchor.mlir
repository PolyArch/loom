// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_anchor_baseline.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ANCHOR
// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MCS

// Acceptance criterion 1 (mcs): on a tier-A workload (all inputs
// isomorphic), mcs produces a FU whose CostModel score is `<=` the
// anchor strategy's score on the same input.
//
// Two arith.addi/subi subgraphs of identical topology and width 32
// share one fabric.op{op_list=[addi,subi]}. Both strategies converge
// on the same single-op FU; the CostModel evaluates that wrapper at
// `1.0` (one fabric.op of width 32 at baseUnit=1.0 for the addi/subi
// share group). Asserting equality of the magnitudes verifies the
// `<=` bound.

// ANCHOR: synth-stat group=alu_int_32 strategy=anchor reason=success
// ANCHOR-SAME: cost=1.000000e+00

// MCS: synth-stat group=alu_int_32 strategy=mcs reason=success
// MCS-SAME: cost=1.000000e+00
// MCS: func.func @fu_alu_int_32
// MCS: fabric.fu
// MCS: fabric.op [@arith.addi, @arith.subi]

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
