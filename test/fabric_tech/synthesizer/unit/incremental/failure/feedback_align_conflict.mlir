// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/../incremental.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier C feedback_align_conflict: two reductive accumulators driven by
// dataflow.streams whose `cont_cond` disagrees ("<" vs ">"). The flow-
// signature heuristic refuses to merge their carry heads (the upstream
// stream signatures differ) and the strategy reports
// `feedback_align_conflict`. Both input functions pick up
// `loom.synth_failed = "feedback_align_conflict"`.

// CHECK: warning:
// CHECK-SAME: group "accum_conflict": synthesis failed: feedback_align_conflict
// CHECK: remark: {{.*}}synth-stat group=accum_conflict strategy=incremental reason=feedback_align_conflict
// CHECK: loom.synth_failed = "feedback_align_conflict"
// CHECK: loom.synth_failed = "feedback_align_conflict"

func.func @pat_accum_lt(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "accum_conflict"} {
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

func.func @pat_accum_gt(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "accum_conflict"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = ">"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.addi %c, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}
