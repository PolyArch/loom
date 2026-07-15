// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// The two stream functions select different correlated attribute tuples.
// Each tuple is a complete hw_params mode and a distinct explicit encoding;
// no field-wise Cartesian product is legal.

// CHECK: remark: {{.*}}synth-stat group=stream_axes strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=2
// CHECK: fabric.module @fu_stream_axes
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.stream]
// CHECK-SAME: hw_params = [
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
