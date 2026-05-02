// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_cap.yaml dump-stats=true' 2>&1 | FileCheck %s

// A workload that has no single local MCES candidate needs compatibility
// search. With candidate_cap=1, launching that search would exceed the
// remaining candidate budget, so mcs reports resource exhaustion.

// CHECK: warning:
// CHECK-SAME: group "alu_int_32": synthesis failed: resource_exhausted
// CHECK: remark: {{.*}}synth-stat group=alu_int_32 strategy=mcs reason=resource_exhausted
// CHECK: loom.synth_failed = "resource_exhausted"
// CHECK: loom.synth_failed = "resource_exhausted"

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_muli(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.muli %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
