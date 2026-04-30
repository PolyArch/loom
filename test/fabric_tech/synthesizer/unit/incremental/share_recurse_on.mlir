// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/share_recurse_on.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier-B input where the diff "extra branch" sub-op (`arith.subi`)
// shares a hardware share-group (`{arith.addi, arith.subi}`) with the
// `arith.addi` already in the FU body. With
// `synth.subgraph_share_recurse: true`, the Incremental tier-B
// candidate generator emits an extra recursive-compression candidate
// in addition to the standard mux/demux baseline. The recursive
// variant widens the new tail fabric.op's `op_list` to the
// share-group-aware union `[@arith.addi, @arith.subi]`, signalling
// that the synthesized hardware unit reuses the addi/subi share
// group. Cost-rank picks the recursive candidate when the cost is
// not strictly worse than the baseline.
//
// The `<=` cost contract: the share-recurse-on cost must not exceed
// the share-recurse-off baseline locked in by `share_recurse_off.mlir`
// (cost=194). Op-list union within the same share group does not
// change CostModel's per-op base unit, so equality is the expected
// outcome here; the assertion guards against a future cost-formula
// change accidentally penalizing the share-aware variant.

// CHECK: remark: {{.*}}synth-stat group=sr_demo strategy=incremental reason=success
// CHECK-SAME: cost=1.940000e+02
// CHECK: func.func @fu_sr_demo
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi]
// CHECK: fabric.demux
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK: fabric.mux
// CHECK: fabric.yield

func.func @sr_pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sr_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    dataflow.yield %t : i32
  }
  return %r : i32
}

func.func @sr_pat_addi_then_subi(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "sr_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    %u = arith.subi %t, %z : i32
    dataflow.yield %u : i32
  }
  return %r : i32
}
