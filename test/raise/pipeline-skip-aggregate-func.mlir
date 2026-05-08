// RUN: loom-raise-opt --loom-llvm-func-to-func --loom-llvm-cf-to-cf --loom-llvm-arith-to-arith %s | FileCheck %s

// Two `llvm.func` ops side by side: one has a builtin-only signature
// and is raised to func.func; the other has an LLVM aggregate (struct)
// in its signature and is SKIPPED. Cf-to-cf and arith-to-arith are
// nested under func.func so the skipped llvm.func keeps its body in
// pristine LLVM form (`llvm.br` and `llvm.add` survive untouched).

// CHECK-LABEL: llvm.func @aggregate_kept
// CHECK-NOT: cf.br
// CHECK-NOT: arith.add
// CHECK: llvm.br
// CHECK: llvm.add
// CHECK: llvm.return
llvm.func @aggregate_kept(%arg: !llvm.struct<(i32, i32)>) -> i32 {
    %z = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1
  ^bb1:
    %a = llvm.add %z, %z : i32
    llvm.return %a : i32
}

// CHECK-LABEL: func.func @builtin_raised
// CHECK-NOT: llvm.br
// CHECK: cf.br
// CHECK: arith.addi
// CHECK: return
llvm.func @builtin_raised(%a: i32, %b: i32) -> i32 {
    llvm.br ^bb1
  ^bb1:
    %s = llvm.add %a, %b : i32
    llvm.return %s : i32
}
