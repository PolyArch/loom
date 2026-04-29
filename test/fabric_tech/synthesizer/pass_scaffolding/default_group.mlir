// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true' 2>&1 | FileCheck %s

// Two func.funcs with no `loom.synth_group` attribute land in the implicit
// `default` group. The factory falls back to the `incremental_random` stub
// which immediately reports `topology_mismatch`; both inputs end up
// annotated with `loom.synth_failed = "topology_mismatch"` and the
// `dump-stats=true` flag emits one canonical `synth-stat` line per group.

// CHECK: warning:
// CHECK-SAME: group "default": synthesis failed: topology_mismatch
// CHECK: remark:
// CHECK-SAME: synth-stat group=default strategy=incremental_random reason=topology_mismatch cost=0.000000e+00 covered=0/2 nodes=0/0/0
// CHECK: loom.synth_failed = "topology_mismatch"
// CHECK: loom.synth_failed = "topology_mismatch"

func.func @pat_addi(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
func.func @pat_subi(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
