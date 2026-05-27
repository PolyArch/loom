// RUN: loom-raise-opt --loom-lower-graph-constants %s | FileCheck %s

// Positive case: a graph.func body with a dataflow.stream + an
// arith.constant whose result feeds a downstream streaming-primitive
// op (here a dataflow.invariant) is rewritten to a dataflow.constant
// driven by the body's leading thread_ctrl block argument. Every
// downstream user is rewritten to consume the dataflow.constant.

// CHECK-LABEL: dataflow.graph.func private @g_constant_promoted
// CHECK-NOT: arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[DCONST:.*]] = dataflow.constant %arg0 {const_value = 1.000000e+00 : f32} : f32
// CHECK-DAG: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[INV:.*]] = dataflow.invariant %[[RWC]], %[[DCONST]] : f32
// CHECK: arith.mulf {{.*}}, %[[INV]]
dataflow.graph.func private @g_constant_promoted(%arg0: none, %arg1: i64,
                                                 %arg2: i64, %arg3: i64,
                                                 %arg4: f32) -> (none, f32) {
  %cst = arith.constant 1.000000e+00 : f32
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.invariant %rwc, %cst : f32
  %1 = dataflow.carry %rwc, %arg4, %2 : f32
  %2 = arith.mulf %1, %0 : f32
  dataflow.graph.return %arg0, %1 : none, f32
}

// Negative-bail #1: a graph.func body whose arith.constant has no
// streaming-primitive users (only consumed by another arith op that
// feeds a non-streaming sink) is left untouched -- the per-constant
// "feeds streaming primitive" check rejects pure scalar plumbing.

// CHECK-LABEL: dataflow.graph.func private @g_no_streaming_user
// CHECK: arith.constant 2.000000e+00 : f32
// CHECK-NOT: dataflow.constant
dataflow.graph.func private @g_no_streaming_user(%arg0: none,
                                                 %arg1: f32) -> (none, f32) {
  %cst = arith.constant 2.000000e+00 : f32
  %0 = arith.addf %arg1, %cst : f32
  dataflow.graph.return %arg0, %0 : none, f32
}
