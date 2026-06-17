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

// A negative-step reduction keeps the signed step operand and flips only the
// stream continuation predicate so descending loops execute the original trip
// count instead of terminating after the init token.

// CHECK-LABEL: dataflow.graph.func private @g_desc_red
// CHECK: %[[STEP:.*]] = arith.constant -1 : i64
// CHECK: %[[IDX:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %[[STEP]]
// CHECK-SAME: cont_cond = ">"
// CHECK-SAME: step_op = "+="
// CHECK: %[[CARRY:.*]] = dataflow.carry %[[RWC]], %arg4,
// CHECK-NOT: scf.for
// CHECK-NOT: scf.yield
// CHECK: dataflow.graph.return %arg0, %[[CARRY]] : none, f32
dataflow.graph.func private @g_desc_red(%ctrl: none, %lb: i64, %ub: i64,
                                        %buf: !llvm.ptr,
                                        %init: f32) -> (none, f32) {
  %c-1_i64 = arith.constant -1 : i64
  %r = scf.for %i = %lb to %ub step %c-1_i64 iter_args(%acc = %init) -> (f32) : i64 {
    %p = llvm.getelementptr %buf[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %v = llvm.load %p : !llvm.ptr -> f32
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  dataflow.graph.return %ctrl, %r : none, f32
}

// Pointer-carried reductions must not let the false stream sentinel drive
// address-generation ops. The pointer value consumed by the lifted body is
// routed through dataflow.gate, while non-pointer loop-carried values keep
// the plain carry output.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carried_red
// CHECK: %[[IDX:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[PTR_CARRY:.*]] = dataflow.carry %[[RWC]], %arg4,
// CHECK: %{{.*}}, %[[PTR_BODY:.*]] = dataflow.gate %[[RWC]], %[[PTR_CARRY]] : !llvm.ptr
// CHECK: %[[ACC_CARRY:.*]] = dataflow.carry %[[RWC]], %arg5,
// CHECK: llvm.getelementptr %[[PTR_BODY]]
// CHECK-NOT: scf.for
// CHECK: dataflow.graph.return %arg0, %[[ACC_CARRY]] : none, i32
dataflow.graph.func private @g_pointer_carried_red(%ctrl: none, %lb: i32,
                                                   %ub: i32, %step: i32,
                                                   %buf: !llvm.ptr,
                                                   %init: i32)
    -> (none, i32) {
  %r:2 = scf.for %i = %lb to %ub step %step
      iter_args(%ptr = %buf, %acc = %init) -> (!llvm.ptr, i32) : i32 {
    %next = llvm.getelementptr %ptr[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %v = llvm.load %next : !llvm.ptr -> i32
    %s = arith.addi %acc, %v : i32
    scf.yield %next, %s : !llvm.ptr, i32
  }
  dataflow.graph.return %ctrl, %r#1 : none, i32
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
