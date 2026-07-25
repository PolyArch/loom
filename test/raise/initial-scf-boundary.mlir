// RUN: split-file %s %t
// RUN: %loom-raise %t/parallel-store.ll | FileCheck %s --check-prefix=STANDARD
// RUN: %loom-raise %t/parallel-store.ll | loom-raise-opt --loom-scf-for-to-forall | FileCheck %s --check-prefix=EXPLICIT

// STANDARD-LABEL: llvm.func @parallel_store
// STANDARD-NOT: scf.forall
// STANDARD: scf.for %
// STANDARD-NOT: scf.forall
// STANDARD: llvm.return

// EXPLICIT-LABEL: llvm.func @parallel_store
// EXPLICIT: scf.forall

//--- parallel-store.ll
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
