// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/../incremental.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier C is triggered by feedback/state in the new input. An already
// stateful FU must not cause a later acyclic input to be grafted through
// the Tier-C mirror builder.

// CHECK: warning:
// CHECK-SAME: group "state_to_acyclic": synthesis failed: topology_mismatch
// CHECK: remark: {{.*}}synth-stat group=state_to_acyclic strategy=incremental reason=topology_mismatch
// CHECK: loom.synth_failed = "topology_mismatch"
// CHECK: loom.synth_failed = "topology_mismatch"

func.func @pat_stateful_first(%lb: i32, %ub: i32, %step: i32,
                              %init: i32) -> i32
    attributes {loom.synth_group = "state_to_acyclic"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.addi %c, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}

func.func @pat_acyclic_later(%lb: i32, %ub: i32, %step: i32,
                             %init: i32) -> i32
    attributes {loom.synth_group = "state_to_acyclic"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %out = arith.muli %in, %s : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
