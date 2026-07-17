// RUN: loom-raise-opt --loom-lower-graph-invariant %s | FileCheck %s

// Positive case: a graph.func body with one directly owned stream and a
// scalar block argument keeps the invariant output in the parent domain and
// projects the value through dataflow.gate before body arithmetic.

// CHECK-LABEL: dataflow.graph.func private @g_scalar_invariant
// CHECK: %[[IV:.*]], %[[PHASE:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[INV:.*]] = dataflow.invariant %[[PHASE]], %arg5 : f32
// CHECK: %{{.*}}, %[[BODY_INV:.*]] = dataflow.gate %[[PHASE]], %[[INV]] : f32
// CHECK: %[[CARRY:.*]] = dataflow.carry %[[PHASE]], %{{.*}}
// CHECK: arith.mulf %[[BODY_INV]]
dataflow.graph.func private @g_scalar_invariant(%arg0: none, %arg1: i64,
                                                %arg2: i64, %arg3: i64,
                                                %arg4: f32, %arg5: f32)
    -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
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
// CHECK: %[[IV:.*]], %[[PHASE:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[STEP_RAW:.*]] = dataflow.invariant %[[PHASE]], %arg3 : i64
// CHECK: %{{.*}}, %[[STEP:.*]] = dataflow.gate %[[PHASE]], %[[STEP_RAW]] : i64
// CHECK: arith.addi %{{.*}}, %[[STEP]] : i64
dataflow.graph.func private @g_step_reused(%arg0: none, %arg1: i64,
                                           %arg2: i64, %arg3: i64,
                                           %arg4: i64)
    -> (none, i64) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %0 = dataflow.carry %rwc, %arg4, %1 : i64
  %1 = arith.addi %0, %arg3 : i64
  dataflow.graph.return %arg0, %0 : none, i64
}

// Negative-bail: a graph.func body without any dataflow.stream is
// left untouched -- the pass needs an existing phase to drive new
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

// A direct graph result is already in the exact-one result domain. It does
// not need invariant replay or body projection.

// CHECK-LABEL: dataflow.graph.func private @g_return_passthrough_only
// CHECK: dataflow.stream
// CHECK-NOT: dataflow.invariant
// CHECK-NOT: dataflow.gate
// CHECK: dataflow.graph.return %arg0, %arg4 : none, i32
dataflow.graph.func private @g_return_passthrough_only(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: i32,
    %arg4: i32) -> (none, i32) {
  %iv, %phase = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i32
  dataflow.graph.return %arg0, %arg4 : none, i32
}

// When an argument has both a body use and a direct graph result, only the
// body use is projected through invariant and gate.

// CHECK-LABEL: dataflow.graph.func private @g_body_and_return_use
// CHECK: %[[IV_MIXED:.*]], %[[PHASE_MIXED:.*]] = dataflow.stream
// CHECK: %[[INV_MIXED:.*]] = dataflow.invariant %[[PHASE_MIXED]], %arg4 : i32
// CHECK: %{{.*}}, %[[BODY_MIXED:.*]] = dataflow.gate %[[PHASE_MIXED]], %[[INV_MIXED]] : i32
// CHECK: %[[SUM_MIXED:.*]] = arith.addi %[[IV_MIXED]], %[[BODY_MIXED]] : i32
// CHECK: dataflow.graph.return %arg0, %arg4, %[[SUM_MIXED]] : none, i32, i32
dataflow.graph.func private @g_body_and_return_use(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: i32,
    %arg4: i32) -> (none, i32, i32) {
  %iv, %phase = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i32
  %sum = arith.addi %iv, %arg4 : i32
  dataflow.graph.return %arg0, %arg4, %sum : none, i32, i32
}

// Multiple directly owned streams are legal when their only block-argument
// use is a direct graph result. The pass must not diagnose ambiguity or
// rewrite the passthrough.

// CHECK-LABEL: dataflow.graph.func private @g_multiple_streams_return_passthrough
// CHECK-COUNT-2: dataflow.stream
// CHECK-NOT: dataflow.invariant
// CHECK-NOT: dataflow.gate
// CHECK: dataflow.graph.return %arg0, %arg5 : none, i64
dataflow.graph.func private @g_multiple_streams_return_passthrough(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: i64, %arg5: i64) -> (none, i64) {
  %iv0, %phase0 = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %iv1, %phase1 = dataflow.stream %arg1, %arg4, %arg3
      step add while slt : i64
  dataflow.graph.return %arg0, %arg5 : none, i64
}
