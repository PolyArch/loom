// RUN: split-file %s %t
// RUN: %loom-raise %t/lazy-nested-poison.ll | FileCheck %s --implicit-check-not=arith.andi

// The inner condition is evaluated only after the outer branch selects its
// edge. Combining both conditions would consume poison when the outer
// condition is false.
// CHECK-LABEL: llvm.func @lazy_nested_poison
// CHECK: %[[POISON:.*]] = llvm.mlir.poison : i1
// CHECK: scf.if %arg0
// CHECK: scf.if %[[POISON]]

//--- lazy-nested-poison.ll
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define i32 @lazy_nested_poison(i1 %outer) {
entry:
  br i1 %outer, label %inner, label %exit

inner:
  br i1 poison, label %taken, label %exit

taken:
  ret i32 1

exit:
  ret i32 0
}
