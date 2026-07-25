// RUN: loom-raise-opt --loom-llvm-cf-to-cf %s | FileCheck %s

// Verify llvm.br / llvm.cond_br rewriting inside an imported callable body.
// The pass rewrites the callable region where it stands, so the llvm.func
// keeps its ABI envelope. llvm.return is deliberately preserved: it is the
// return of an imported LLVM callable and the CFG-to-SCF transformation
// treats it as an ordinary return-like exit.

// CHECK-LABEL: llvm.func @ranged_pick
// CHECK-NOT: llvm.br
// CHECK-NOT: llvm.cond_br
// CHECK: cf.br
// CHECK: cf.cond_br
// CHECK: llvm.return
llvm.func @ranged_pick(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1
  ^bb1:
    %cond = llvm.icmp "slt" %arg0, %arg1 : i32
    llvm.cond_br %cond, ^bb2, ^bb3
  ^bb2:
    %sum = llvm.add %arg0, %arg2 : i32
    llvm.return %sum : i32
  ^bb3:
    %diff = llvm.sub %arg1, %arg2 : i32
    llvm.return %diff : i32
}
