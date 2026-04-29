// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A with cross share groups (arith.addi vs arith.muli; muli is in
// a different multi-member group, so the two cannot share one fabric.op
// op_list). Default anchor config has `allow_intra_position_mux=false`,
// so the strategy fails with `cross_share_group` and every input
// function picks up `loom.synth_failed = "cross_share_group"`.

// CHECK: warning:
// CHECK-SAME: group "alu_int_32_x": synthesis failed: cross_share_group
// CHECK: remark: {{.*}}synth-stat group=alu_int_32_x strategy=anchor reason=cross_share_group cost=0.000000e+00 covered=0/2 nodes=0/0/0
// CHECK: loom.synth_failed = "cross_share_group"
// CHECK: loom.synth_failed = "cross_share_group"

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32_x"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_muli(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32_x"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.muli %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
