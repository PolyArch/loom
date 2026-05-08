// RUN: loom-raise-opt --loom-scf-for-to-forall %s | FileCheck %s

// A loop body that contains a volatile store MUST NOT lift to
// scf.forall. Volatile semantics forbid reordering / cloning, which
// scf.forall semantics implicitly allow.

// CHECK-LABEL: llvm.func @volatile_store_kept
// CHECK: scf.for
// CHECK-NOT: scf.forall
llvm.func @volatile_store_kept(%base: !llvm.ptr) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %fzero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c64 step %c1 : i64 {
      %p = llvm.getelementptr inbounds %base[%i]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      llvm.store volatile %fzero, %p : f32, !llvm.ptr
    }
    llvm.return
}

// A non-volatile store with a disjoint base passes for sanity --
// the previous case fails specifically because of the volatile bit.

// CHECK-LABEL: llvm.func @non_volatile_lifts
// CHECK: scf.forall
// CHECK-NOT: scf.for
llvm.func @non_volatile_lifts(%base: !llvm.ptr) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %fzero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c64 step %c1 : i64 {
      %p = llvm.getelementptr inbounds %base[%i]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      llvm.store %fzero, %p : f32, !llvm.ptr
    }
    llvm.return
}
