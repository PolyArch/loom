// RUN: loom-raise-opt --loom-lower-graph-invariant %s | FileCheck %s

// Positive case: a graph.func body with a dataflow.stream + a scalar
// f32 block argument used inside the body wraps that block argument
// in a dataflow.invariant carrier driven by the stream's rwc.

// CHECK-LABEL: dataflow.graph.func private @g_scalar_invariant
// CHECK: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[INV:.*]] = dataflow.invariant %[[RWC]], %arg5 : f32
// CHECK: %[[CARRY:.*]] = dataflow.carry %[[RWC]], %{{.*}}
// CHECK: arith.mulf %[[INV]]
dataflow.graph.func private @g_scalar_invariant(%arg0: none, %arg1: i64,
                                                %arg2: i64, %arg3: i64,
                                                %arg4: f32, %arg5: f32)
    -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.carry %rwc, %arg4, %2 : f32
  %1 = arith.mulf %arg5, %0 : f32
  %2 = arith.addf %0, %1 : f32
  dataflow.graph.return %arg0, %0 : none, f32
}

// Stream-bound args keep their raw uses on dataflow.stream itself,
// but non-stream uses of the same arg are loop-invariant scalar data
// and must be wrapped. This covers reductions whose carried induction
// value advances by the same step that initializes the stream.

// CHECK-LABEL: dataflow.graph.func private @g_step_reused
// CHECK: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[STEP:.*]] = dataflow.invariant %[[RWC]], %arg3 : i64
// CHECK: arith.addi %{{.*}}, %[[STEP]] : i64
dataflow.graph.func private @g_step_reused(%arg0: none, %arg1: i64,
                                           %arg2: i64, %arg3: i64,
                                           %arg4: i64)
    -> (none, i64) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.carry %rwc, %arg4, %1 : i64
  %1 = arith.addi %0, %arg3 : i64
  dataflow.graph.return %arg0, %0 : none, i64
}

// Negative-bail: a graph.func body without any dataflow.stream is
// left untouched -- the pass needs an existing rwc to drive new
// invariant carriers.

// CHECK-LABEL: dataflow.graph.func private @g_no_stream
// CHECK-NOT: dataflow.invariant
// CHECK: scf.for
dataflow.graph.func private @g_no_stream(%arg0: none, %arg1: i64,
                                         %arg2: i64, %arg3: i64,
                                         %arg4: f32, %arg5: f32)
    -> (none, f32) {
  %r = scf.for %i = %arg1 to %arg2 step %arg3 iter_args(%acc = %arg4) -> (f32)
      : i64 {
    %s = arith.addf %acc, %arg5 : f32
    scf.yield %s : f32
  }
  dataflow.graph.return %arg0, %r : none, f32
}
