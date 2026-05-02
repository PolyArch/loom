// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/scc_full_unroll.yaml dump-stats=true' 2>&1 | FileCheck %s --check-prefix=FULL --implicit-check-not=unrealized_conversion_cast

// The default flow-signature heuristic rejects incompatible carry
// signatures. Enabling scc_full_unroll should keep both state paths and
// synthesize one FU that covers the group.

// DEFAULT: warning:
// DEFAULT-SAME: group "full_unroll_state": synthesis failed: feedback_align_conflict
// DEFAULT: remark: {{.*}}synth-stat group=full_unroll_state strategy=incremental reason=feedback_align_conflict
// DEFAULT: loom.synth_failed = "feedback_align_conflict"

// FULL: remark: {{.*}}synth-stat group=full_unroll_state strategy=incremental reason=success
// FULL-SAME: covered=2/2
// FULL: fabric.module @fu_full_unroll_state
// FULL: fabric.pe [spatial]
// FULL: fabric.fu
// FULL: fabric.op [@dataflow.stream]
// FULL: fabric.op [@dataflow.stream]
// FULL: fabric.op [@dataflow.carry]
// FULL: fabric.op [@dataflow.carry]
// FULL-NOT: fabric.op [@dataflow.carry]
// FULL-DAG: fabric.op [@arith.addi]
// FULL-DAG: fabric.op [@arith.xori]
// FULL-DAG: fabric.mux
// FULL-NOT: fabric.op [@dataflow.carry]
// FULL: fabric.yield

func.func @pat_full_unroll_forward(%lb: i32, %ub: i32, %step: i32,
                                   %init: i32) -> i32
    attributes {loom.synth_group = "full_unroll_state"} {
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

func.func @pat_full_unroll_reverse(%lb: i32, %ub: i32, %step: i32,
                                   %init: i32) -> i32
    attributes {loom.synth_group = "full_unroll_state"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "-=", cont_cond = ">"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.xori %c, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}
