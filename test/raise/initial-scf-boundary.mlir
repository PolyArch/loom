// RUN: split-file %s %t
// RUN: %loom-raise %t/parallel-store.ll | FileCheck %s --check-prefix=STANDARD
// RUN: %loom-raise %t/parallel-store.ll | loom-raise-opt --loom-scf-for-to-forall | FileCheck %s --check-prefix=EXPLICIT

// The source loop is post-tested: its exit comparison observes the
// already-bumped induction value, so mechanical raising cannot prove the
// scf.for trip count and preserves the recovered serial loop as scf.while.
// A serial loop remains serial until a typed decision transforms it. The
// counted parallel decision consumes scf.for, so it has no candidate here:
// the loop stays serial scf.while even when the decision pass runs
// explicitly.

// STANDARD-LABEL: llvm.func @parallel_store
// STANDARD-NOT: scf.forall
// STANDARD: scf.while
// STANDARD-NOT: scf.forall
// STANDARD: llvm.return

// EXPLICIT-LABEL: llvm.func @parallel_store
// EXPLICIT-NOT: scf.forall
// EXPLICIT: scf.while
// EXPLICIT-NOT: scf.forall
// EXPLICIT: llvm.return

//--- parallel-store.ll
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define void @parallel_store(ptr %output) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %address = getelementptr i32, ptr %output, i64 %iv
  store i32 0, ptr %address, align 4
  %next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
