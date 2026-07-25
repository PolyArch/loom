// RUN: loom-raise-opt --loom-scf-for-to-forall %s | FileCheck %s

// A loop whose body contains a func.call to an unmodelled callee MUST
// NOT lift to scf.forall: the callee may have side effects we cannot
// reason about safely across parallel iterations.

func.func private @opaque_callee(%i: index) -> f32

// CHECK-LABEL: func.func @opaque_call_in_body
// CHECK: scf.for
// CHECK-NOT: scf.forall
func.func @opaque_call_in_body(%dst: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = func.call @opaque_callee(%i) : (index) -> f32
      memref.store %v, %dst[%i] : memref<?xf32>
    }
    return
}

// llvm.call to an unmodelled callee inside the body is also a
// bail-out: the callee's side effects are opaque.

llvm.func @opaque_llvm_callee(i64) -> f32

// CHECK-LABEL: llvm.func @llvm_call_in_body
// CHECK: scf.for
// CHECK-NOT: scf.forall
llvm.func @llvm_call_in_body(%base: !llvm.ptr) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    scf.for %i = %c0 to %c64 step %c1 : i64 {
      %v = llvm.call @opaque_llvm_callee(%i)
        : (i64) -> f32
      %p = llvm.getelementptr inbounds %base[%i]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      llvm.store %v, %p : f32, !llvm.ptr
    }
    llvm.return
}

// A nested callable is a definition, not an operation executed by the outer
// loop. Its body belongs to that callable and must not make an otherwise
// parallel outer loop fail the body check.
// CHECK-LABEL: func.func @nested_callable_is_not_loop_body
// CHECK: scf.forall
// CHECK: module {
// CHECK: func.func @inner
func.func @nested_callable_is_not_loop_body(%dst: memref<?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    builtin.module {
      func.func private @opaque()
      func.func @inner() {
        func.call @opaque() : () -> ()
        return
      }
    }
    memref.store %f0, %dst[%i] : memref<?xf32>
  }
  return
}
