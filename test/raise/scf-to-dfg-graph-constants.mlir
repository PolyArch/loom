// RUN: loom-raise-opt --loom-lower-graph-constants %s | FileCheck %s

// Positive case: a graph body with a dataflow.stream + an
// arith.constant whose result feeds a downstream streaming-primitive
// op (here a dataflow.invariant) is rewritten to a dataflow.constant
// driven by the body's leading thread_ctrl block argument. Every
// downstream user is rewritten to consume the dataflow.constant.

// CHECK-LABEL: dataflow.graph private @g_constant_promoted
// CHECK-NOT: arith.constant 1.000000e+00 : f32
// CHECK-DAG: %[[DCONST:.*]] = dataflow.constant %arg0 {const_value = 1.000000e+00 : f32} : f32
// CHECK-DAG: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[INV:.*]] = dataflow.invariant %[[RWC]], %[[DCONST]] : f32
// CHECK: arith.mulf {{.*}}, %[[INV]]
dataflow.graph private @g_constant_promoted(%arg0: none, %arg1: i64,
                                                 %arg2: i64, %arg3: i64,
                                                 %arg4: f32) -> (f32) {
  %cst = arith.constant 1.000000e+00 : f32
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %0 = dataflow.invariant %rwc, %cst : f32
  %1 = dataflow.carry %rwc, %arg4, %2 : f32
  %2 = arith.mulf %1, %0 : f32
  dataflow.graph.return %arg0, %1 : none, f32
}

// Scalar constants inside a graph are also hardware-visible sources even when
// they only feed scalar arithmetic. They must lower to dataflow.constant so
// PnR sees a real configurable constant resource instead of a residual
// arith.constant op.

// CHECK-LABEL: dataflow.graph private @g_no_streaming_user
// CHECK-NOT: arith.constant 2.000000e+00 : f32
// CHECK: %[[SCALAR_CONST:.*]] = dataflow.constant %arg0 {const_value = 2.000000e+00 : f32} : f32
// CHECK: arith.addf %arg1, %[[SCALAR_CONST]]
dataflow.graph private @g_no_streaming_user(%arg0: none,
                                                 %arg1: f32) -> (f32) {
  %cst = arith.constant 2.000000e+00 : f32
  %0 = arith.addf %arg1, %cst : f32
  dataflow.graph.return %arg0, %0 : none, f32
}

// Identical graph-start constants are one actor with multicast consumers. A
// cloned lane must not consume another resident TokenControl context.

// CHECK-LABEL: dataflow.graph private @g_shared_constant
// CHECK: %[[ZERO:.*]] = dataflow.constant %arg0 {const_value = 0 : i64} : i64
// CHECK-NOT: dataflow.constant %arg0 {const_value = 0 : i64} : i64
// CHECK: arith.addi %arg1, %[[ZERO]] : i64
// CHECK: arith.addi %arg2, %[[ZERO]] : i64
dataflow.graph private @g_shared_constant(%arg0: none, %arg1: i64,
                                          %arg2: i64) -> (i64, i64) {
  %zero0 = arith.constant 0 : i64
  %zero1 = arith.constant 0 : i64
  %lhs = arith.addi %arg1, %zero0 : i64
  %rhs = arith.addi %arg2, %zero1 : i64
  dataflow.graph.return %arg0, %lhs, %rhs : none, i64, i64
}

// Poison is not a literal and cannot be assigned defined bits by constant
// lowering. It remains explicit until the canonical operation schema owns its
// propagation and observation semantics.

// CHECK-LABEL: dataflow.graph private @g_poison_preserved
// CHECK: %[[POISON:.*]] = ub.poison : i32
// CHECK: arith.select %arg1, %[[POISON]], %arg2 : i32
// CHECK-NOT: const_value = 0 : i32
dataflow.graph private @g_poison_preserved(%arg0: none, %arg1: i1,
                                           %arg2: i32) -> (i32) {
  %poison = ub.poison : i32
  %selected = arith.select %arg1, %poison, %arg2 : i32
  dataflow.graph.return %arg0, %selected : none, i32
}
