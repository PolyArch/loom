// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// Tier C: the largest-first order may seed the FU from the stateful
// input, then fold a compatible acyclic peer by muxing the yielded
// post-state and acyclic values.

// CHECK: remark: {{.*}}synth-stat group=fresh_carry strategy=incremental reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_fresh_carry
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@dataflow.stream]
// CHECK-DAG: fabric.op [@dataflow.carry]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.mux
// CHECK: fabric.yield

func.func @pat_fresh_acyclic(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "fresh_carry"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %sum = arith.addi %in, %idx : i32
    dataflow.yield %sum : i32
  }
  return %r : i32
}

func.func @pat_fresh_with_carry(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "fresh_carry"} {
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
