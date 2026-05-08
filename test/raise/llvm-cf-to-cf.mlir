// RUN: loom-raise-opt --loom-llvm-func-to-func --loom-llvm-cf-to-cf %s | FileCheck %s

// Verify llvm.br / llvm.cond_br rewriting inside a function body. The
// cf-to-cf pass is nested under func.func, so we first raise the
// llvm.func into a func.func via loom-llvm-func-to-func and then run
// cf-to-cf to convert the branch terminators. After raising, the
// llvm.return is rewritten to func.return by func-to-func itself.

// CHECK-LABEL: func.func @ranged_pick
// CHECK-NOT: llvm.br
// CHECK-NOT: llvm.cond_br
// CHECK: cf.br
// CHECK: cf.cond_br
// CHECK: return
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
