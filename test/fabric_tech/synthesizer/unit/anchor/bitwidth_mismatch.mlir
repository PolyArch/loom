// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor_with_mux.yaml dump-stats=true' 2>&1 | FileCheck %s

// Two i32 subgraphs and one i64 subgraph at the same topology
// position. Bit-width is part of share-group identity; even with the
// intra-position mux flag enabled, the strategy returns
// `topology_mismatch`.

// CHECK: warning:
// CHECK-SAME: group "alu_int_mixed_bw": synthesis failed: topology_mismatch
// CHECK: remark: {{.*}}synth-stat group=alu_int_mixed_bw strategy=anchor reason=topology_mismatch
// CHECK: loom.synth_failed = "topology_mismatch"
// CHECK: loom.synth_failed = "topology_mismatch"
// CHECK: loom.synth_failed = "topology_mismatch"

func.func @pat_addi_i32_a(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_mixed_bw"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_addi_i32_b(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_mixed_bw"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_addi_i64(%a: i64, %b: i64) -> i64
    attributes {loom.synth_group = "alu_int_mixed_bw"} {
  %r = dataflow.subgraph(%x = %a : i64, %y = %b : i64) -> i64 {
    %s = arith.addi %x, %y : i64
    dataflow.yield %s : i64
  }
  return %r : i64
}
