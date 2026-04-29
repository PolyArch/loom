// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_timeout.yaml dump-stats=true' 2>&1 | FileCheck %s

// `mcs.timeout_sec=0` disables the strategy's wall-time budget. Mcs
// short-circuits before launching any branch and reports `timeout`,
// the spec's failure code for "a strategy exceeded its `timeout_sec`
// budget". This deterministically exercises the timeout path without
// needing a pathological wall-clock-sensitive workload.

// CHECK: warning:
// CHECK-SAME: group "alu_int_32": synthesis failed: timeout
// CHECK: remark: {{.*}}synth-stat group=alu_int_32 strategy=mcs reason=timeout
// CHECK: loom.synth_failed = "timeout"
// CHECK: loom.synth_failed = "timeout"

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
