// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A: two subgraphs of identical topology. Each yields the index port
// of a `dataflow.stream` whose `step_op` and `cont_cond` attributes
// differ between inputs (`+=`/`<` and `-=`/`>`). The synthesized FU's
// hw_params must surface the observed-value union of step_op and
// cont_cond, sorted lexically. Without the union the enumerator's
// step_op/cont_cond axes would not fan out and coverage would fail.

// CHECK: remark: {{.*}}synth-stat group=stream_axes strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: func.func @fu_stream_axes
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.stream]
// CHECK-SAME: hw_params = [{cont_cond = ["<", ">"], step_op = ["+=", "-="]}]
// CHECK: fabric.yield

func.func @pat_stream_inc_lt(%lb: i32, %ub: i32, %step: i32) -> i32
    attributes {loom.synth_group = "stream_axes"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32)
                       -> i32 {
    %i, %rwc = dataflow.stream %l, %u, %s
                {step_op = "+=", cont_cond = "<"} : i32
    dataflow.yield %i : i32
  }
  return %r : i32
}

func.func @pat_stream_dec_gt(%lb: i32, %ub: i32, %step: i32) -> i32
    attributes {loom.synth_group = "stream_axes"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32)
                       -> i32 {
    %i, %rwc = dataflow.stream %l, %u, %s
                {step_op = "-=", cont_cond = ">"} : i32
    dataflow.yield %i : i32
  }
  return %r : i32
}
