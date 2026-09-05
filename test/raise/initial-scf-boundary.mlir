// RUN: split-file %s %t
// RUN: %loom-raise %t/parallel-store.ll | FileCheck %s --check-prefix=STANDARD
// RUN: %loom-raise %t/parallel-store.ll | loom-raise-opt --loom-scf-for-to-forall | FileCheck %s --check-prefix=EXPLICIT

// The finite post-tested source loop satisfies the exact counted-loop
// projection and mechanically normalizes to serial scf.for. It remains serial
// until a typed decision transforms it; the explicitly requested counted
// parallel decision then materializes scf.forall.

// STANDARD-LABEL: llvm.func @parallel_store
// STANDARD-NOT: scf.forall
// STANDARD: scf.for
// STANDARD-NOT: scf.forall
// STANDARD: llvm.return

// EXPLICIT-LABEL: llvm.func @parallel_store
// EXPLICIT: scf.forall
// EXPLICIT-NOT: scf.for
// EXPLICIT: llvm.return

//--- parallel-store.ll
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define void @parallel_store(ptr %output) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %address = getelementptr inbounds i32, ptr %output, i64 %iv
  store i32 0, ptr %address, align 4
  %next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
