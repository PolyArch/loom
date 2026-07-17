// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Predicate selection produces complete modes over one fixed add datapath.

// CHECK: remark: {{.*}}synth-stat group=stream_axes strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=2
// CHECK: fabric.module @fu_stream_axes
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-SAME: valid_encodings = [
// CHECK-SAME: {mode = 0 : i32, resource = 0 : i32}
// CHECK-SAME: {mode = 1 : i32, resource = 0 : i32}
// CHECK: fabric.op [@dataflow.stream]
// CHECK-SAME: hw_params = [
// CHECK: attributes = {predicate = 2 : i64, step_kind = 0 : i32}
// CHECK: attributes = {predicate = 4 : i64, step_kind = 0 : i32}
// CHECK: fabric.yield

func.func @pat_stream_inc_lt(%lb: i32, %ub: i32, %step: i32) -> i32
    attributes {loom.synth_group = "stream_axes"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32)
                       -> i32 {
    %i, %rwc = dataflow.stream %l, %u, %s
                step add while slt : i32
    dataflow.yield %i : i32
  }
  return %r : i32
}

func.func @pat_stream_dec_gt(%lb: i32, %ub: i32, %step: i32) -> i32
    attributes {loom.synth_group = "stream_axes"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32)
                       -> i32 {
    %i, %rwc = dataflow.stream %l, %u, %s step add while sgt : i32
    dataflow.yield %i : i32
  }
  return %r : i32
}
