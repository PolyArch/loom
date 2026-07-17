// RUN: loom-raise-opt --loom-lower-reduction-to-stream %s | FileCheck %s

// scf.for with iter_args(f32) inside a dataflow.graph.func body that
// matches the simple-reduction shape (no nested SCF, no calls,
// loop-invariant lb/ub/step) is rewritten into dataflow.stream +
// dataflow.carry plus explicit true-domain body and false-domain result
// projections. The body ops are moved into the graph entry block.

// CHECK-LABEL: dataflow.graph.func private @g_simple_red
// CHECK: %[[IV:.*]], %[[PHASE:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK-SAME: step add while slt
// CHECK: %[[CARRY:.*]] = dataflow.carry %[[PHASE]], %arg5, %[[NEXT:.*]] : f32
// CHECK: %{{.*}}, %[[BODY_CARRY:.*]] = dataflow.gate %[[PHASE]], %[[CARRY]] : f32
// CHECK: %[[EXIT:.*]]:2 = dataflow.demux %[[PHASE]], %[[CARRY]] : (i1, f32) -> (f32, f32)
// CHECK: %[[NEXT]] = arith.addf %[[BODY_CARRY]],
// CHECK-NOT: scf.for
// CHECK-NOT: scf.yield
// CHECK: dataflow.graph.return %arg0, %[[EXIT]]#0 : none, f32
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
// CHECK: %[[IV:.*]], %[[PHASE:.*]] = dataflow.stream %arg1, %arg2, %[[STEP]]
// CHECK-SAME: step add while sgt
// CHECK: %[[CARRY:.*]] = dataflow.carry %[[PHASE]], %arg4, %[[NEXT:.*]] : f32
// CHECK: %{{.*}}, %[[BODY_CARRY:.*]] = dataflow.gate %[[PHASE]], %[[CARRY]] : f32
// CHECK: %[[EXIT:.*]]:2 = dataflow.demux %[[PHASE]], %[[CARRY]] : (i1, f32) -> (f32, f32)
// CHECK: %[[NEXT]] = arith.addf %[[BODY_CARRY]],
// CHECK-NOT: scf.for
// CHECK-NOT: scf.yield
// CHECK: dataflow.graph.return %arg0, %[[EXIT]]#0 : none, f32
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

// Every loop-carried value has distinct parent, body, and exit domains.
// This applies uniformly to pointers and scalar values.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carried_red
// CHECK: %[[IV:.*]], %[[PHASE:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[PTR_CARRY:.*]] = dataflow.carry %[[PHASE]], %arg4,
// CHECK: %{{.*}}, %[[PTR_BODY:.*]] = dataflow.gate %[[PHASE]], %[[PTR_CARRY]] : !llvm.ptr
// CHECK: %[[PTR_EXIT:.*]]:2 = dataflow.demux %[[PHASE]], %[[PTR_CARRY]] : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
// CHECK: %[[ACC_CARRY:.*]] = dataflow.carry %[[PHASE]], %arg5, %[[ACC_NEXT:.*]] : i32
// CHECK: %{{.*}}, %[[ACC_BODY:.*]] = dataflow.gate %[[PHASE]], %[[ACC_CARRY]] : i32
// CHECK: %[[ACC_EXIT:.*]]:2 = dataflow.demux %[[PHASE]], %[[ACC_CARRY]] : (i1, i32) -> (i32, i32)
// CHECK: llvm.getelementptr %[[PTR_BODY]]
// CHECK: %[[ACC_NEXT]] = arith.addi %[[ACC_BODY]],
// CHECK-NOT: scf.for
// CHECK: dataflow.graph.return %arg0, %[[ACC_EXIT]]#0 : none, i32
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
