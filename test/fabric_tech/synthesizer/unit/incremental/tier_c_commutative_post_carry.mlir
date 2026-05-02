// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// Tier C: post-carry users with commutative operand order should merge
// even when one input writes the carried value on the right-hand side.

// CHECK: remark: {{.*}}synth-stat group=comm_post strategy=incremental reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_comm_post
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.stream]
// CHECK: fabric.op [@dataflow.carry]
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.xori]
// CHECK-DAG: fabric.mux
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK: fabric.yield

// CHECK: remark: {{.*}}synth-stat group=feedback_mux3 strategy=incremental reason=success
// CHECK-SAME: covered=3/3
// CHECK: fabric.module @fu_feedback_mux3
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.stream]
// CHECK: fabric.op [@dataflow.carry]
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.xori]
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK-DAG: fabric.mux
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK: fabric.yield

func.func @pat_comm_addi(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "comm_post"} {
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

func.func @pat_comm_xori_swapped(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "comm_post"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.xori %idx, %c : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}

func.func @pat_mux3_addi(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "feedback_mux3"} {
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

func.func @pat_mux3_xori(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "feedback_mux3"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.xori %c, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}

func.func @pat_mux3_muli(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "feedback_mux3"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.muli %c, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}
