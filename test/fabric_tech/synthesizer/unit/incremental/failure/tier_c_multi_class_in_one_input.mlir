// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/../incremental.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier C rejects a same-class duplicate head conflict: one input
// contributes multiple carry heads to the same flow-signature class.
// Both inputs keep the same external shape and result arity so the
// failure is isolated to feedback alignment.

// CHECK: warning:
// CHECK-SAME: group "multi_class_conflict": synthesis failed: feedback_align_conflict
// CHECK: remark: {{.*}}synth-stat group=multi_class_conflict strategy=incremental reason=feedback_align_conflict
// CHECK: loom.synth_failed = "feedback_align_conflict"
// CHECK: loom.synth_failed = "feedback_align_conflict"

func.func @pat_conflict_two_matching_carries(%lb0: i32, %ub0: i32, %step0: i32,
                                             %lb1: i32, %ub1: i32, %step1: i32,
                                             %init0: i32, %init1: i32)
    -> (i32, i32)
    attributes {loom.synth_group = "multi_class_conflict"} {
  %p, %q = dataflow.subgraph(%l0 = %lb0 : i32, %u0 = %ub0 : i32,
                             %s0 = %step0 : i32, %l1 = %lb1 : i32,
                             %u1 = %ub1 : i32, %s1 = %step1 : i32,
                             %in0 = %init0 : i32, %in1 = %init1 : i32)
      -> (i32, i32) {
    %idx0, %rwc0 = dataflow.stream %l0, %u0, %s0
                  {step_op = "+=", cont_cond = "<"} : i32
    %idx1, %rwc1 = dataflow.stream %l1, %u1, %s1
                  {step_op = "+=", cont_cond = "<"} : i32
    %c0 = dataflow.carry %rwc0, %in0, %nxt0 : i32
    %c1 = dataflow.carry %rwc1, %in1, %nxt1 : i32
    %nxt0 = arith.addi %c0, %idx0 : i32
    %nxt1 = arith.xori %c1, %idx1 : i32
    dataflow.yield %c0, %c1 : i32, i32
  }
  return %p, %q : i32, i32
}

func.func @pat_conflict_distinct_carries(%lb0: i32, %ub0: i32, %step0: i32,
                                         %lb1: i32, %ub1: i32, %step1: i32,
                                         %init0: i32, %init1: i32)
    -> (i32, i32)
    attributes {loom.synth_group = "multi_class_conflict"} {
  %p, %q = dataflow.subgraph(%l0 = %lb0 : i32, %u0 = %ub0 : i32,
                             %s0 = %step0 : i32, %l1 = %lb1 : i32,
                             %u1 = %ub1 : i32, %s1 = %step1 : i32,
                             %in0 = %init0 : i32, %in1 = %init1 : i32)
      -> (i32, i32) {
    %idx0, %rwc0 = dataflow.stream %l0, %u0, %s0
                  {step_op = "+=", cont_cond = "<"} : i32
    %idx1, %rwc1 = dataflow.stream %l1, %u1, %s1
                  {step_op = "*=", cont_cond = "<="} : i32
    %c0 = dataflow.carry %rwc0, %in0, %nxt0 : i32
    %c1 = dataflow.carry %rwc1, %in1, %nxt1 : i32
    %nxt0 = arith.addi %c0, %idx0 : i32
    %nxt1 = arith.xori %c1, %idx1 : i32
    dataflow.yield %c0, %c1 : i32, i32
  }
  return %p, %q : i32, i32
}
