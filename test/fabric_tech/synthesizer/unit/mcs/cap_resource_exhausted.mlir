// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_cap.yaml dump-stats=true' 2>&1 | FileCheck %s

// Acceptance criterion 3 (mcs): on a workload reaching `candidate_cap`,
// mcs returns `resource_exhausted`. The cap config (`mcs_cap.yaml`)
// pins `candidate_cap=1`; with two inputs the strategy plans more than
// one branch (anchor branches alone give one per input plus at least
// one random branch), so the planned count strictly exceeds the cap and
// the strategy fails before launching any branch.

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

func.func @pat_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
