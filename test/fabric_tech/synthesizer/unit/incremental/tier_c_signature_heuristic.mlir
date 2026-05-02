// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier C signature-heuristic example: two reductive accumulators driven
// by identical streams (lb=0, ub=N, step=1, step_op="+=", cont_cond="<")
// feed a dataflow.carry whose carried value is post-processed
// differently (arith.addi vs arith.xori). The flow-signature heuristic
// matches both carries by their (carry_type=i32,
// upstream_stream=(i32,"+=","<")) tuple so the carry head merges; the
// post-carry diff is then handled as a tier-B mux insertion behind the
// carry's back-edge operand.

// CHECK: remark: {{.*}}synth-stat group=accum strategy=incremental reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_accum
// CHECK: fabric.pe [spatial]
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
