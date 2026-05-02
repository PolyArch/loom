// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier C feedback alignment: two reductive accumulators driven by
// identical streams (lb=0, ub=N, step=1, step_op="+=", cont_cond="<")
// feed a dataflow.carry whose carried value is post-processed by
// arith.addi vs arith.xori. Same input as the incremental strategy's
// tier-C signature_heuristic test; mcs delegates to the same Tier-C
// trivial-FU + mux-insert path through Incremental and reaches the
// same success outcome.

// CHECK: remark: {{.*}}synth-stat group=accum strategy=mcs reason=success
// CHECK-SAME: covered=2/2
// CHECK: func.func @fu_accum
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@dataflow.stream]
// CHECK-DAG: fabric.op [@dataflow.carry]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.xori]
// CHECK-DAG: fabric.mux
// CHECK: fabric.yield

func.func @pat_accum_addi(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "accum"} {
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

func.func @pat_accum_xori(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "accum"} {
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
