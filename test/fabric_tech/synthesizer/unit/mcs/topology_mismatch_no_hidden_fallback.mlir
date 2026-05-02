// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_cap.yaml dump-stats=true' 2>&1 | FileCheck %s

// A workload with no graph-native MCES candidate now fails directly in MCS.
// Fallback policy belongs to the outer fallback_chain, not to a hidden
// compatibility branch inside the strategy.

// CHECK: warning:
// CHECK-SAME: group "alu_int_32": synthesis failed: topology_mismatch
// CHECK: remark: {{.*}}synth-stat group=alu_int_32 strategy=mcs reason=topology_mismatch
// CHECK: loom.synth_failed = "topology_mismatch"
// CHECK: loom.synth_failed = "topology_mismatch"

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
