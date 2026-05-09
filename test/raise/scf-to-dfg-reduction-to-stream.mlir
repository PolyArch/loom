// RUN: loom-raise-opt --loom-lower-reduction-to-stream %s | FileCheck %s

// scf.for with iter_args(f32) inside a dataflow.graph.func body that
// matches the simple-reduction shape (no nested SCF, no calls,
// loop-invariant lb/ub/step) is rewritten into dataflow.stream +
// dataflow.carry plus the body ops moved out into the graph entry
// block. The graph.return continues to feed the carry's output.

// CHECK-LABEL: dataflow.graph.func private @g_simple_red
// CHECK: %[[IDX:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK-SAME: cont_cond = "<"
// CHECK-SAME: step_op = "+="
// CHECK: %[[CARRY:.*]] = dataflow.carry %[[RWC]], %arg5,
// CHECK-NOT: scf.for
// CHECK-NOT: scf.yield
// CHECK: dataflow.graph.return %arg0, %[[CARRY]] : none, f32
dataflow.graph.func private @g_simple_red(%ctrl: none, %lb: i64, %ub: i64,
                                          %step: i64, %buf: !llvm.ptr,
                                          %init: f32) -> (none, f32) {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i64 {
    %p = llvm.getelementptr %buf[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %v = llvm.load %p : !llvm.ptr -> f32
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  dataflow.graph.return %ctrl, %r : none, f32
}

// Negative-bail #1: a graph.func with a nested scf.for in its body is
// left unchanged. The pass emits a remark; the loop survives as-is.

// CHECK-LABEL: dataflow.graph.func private @g_nested_for
// CHECK: scf.for %{{.*}} iter_args(%[[ACC:.*]] = %arg5)
// CHECK:   scf.for %{{.*}}
// CHECK-NOT: dataflow.stream
// CHECK-NOT: dataflow.carry
dataflow.graph.func private @g_nested_for(%ctrl: none, %lb: i64, %ub: i64,
                                          %step: i64, %buf: !llvm.ptr,
                                          %init: f32) -> (none, f32) {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i64 {
    %s = scf.for %j = %lb to %ub step %step iter_args(%inner = %acc) -> (f32) : i64 {
      %p = llvm.getelementptr %buf[%j] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %v = llvm.load %p : !llvm.ptr -> f32
      %t = arith.addf %inner, %v : f32
      scf.yield %t : f32
    }
    scf.yield %s : f32
  }
  dataflow.graph.return %ctrl, %r : none, f32
}

// Negative-bail #2: a graph.func with an `llvm.call` inside the for
// body is left unchanged. The pass bails on any CallOpInterface op.
// (We use `llvm.call` rather than `func.call` because the graph.func
// body verifier rejects `func.call` directly, and `llvm.call`
// implements the same CallOpInterface that the bail logic checks.)

// CHECK-LABEL: dataflow.graph.func private @g_with_call
// CHECK: scf.for %{{.*}} iter_args
// CHECK:   llvm.call @sink
// CHECK-NOT: dataflow.stream
// CHECK-NOT: dataflow.carry
llvm.func @sink(f32) -> f32

dataflow.graph.func private @g_with_call(%ctrl: none, %lb: i64, %ub: i64,
                                         %step: i64, %buf: !llvm.ptr,
                                         %init: f32) -> (none, f32) {
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i64 {
    %p = llvm.getelementptr %buf[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %v = llvm.load %p : !llvm.ptr -> f32
    %w = llvm.call @sink(%v) : (f32) -> f32
    %s = arith.addf %acc, %w : f32
    scf.yield %s : f32
  }
  dataflow.graph.return %ctrl, %r : none, f32
}
