// RUN: loom-raise-opt --loom-llvm-cf-to-cf --loom-llvm-func-to-func %s | FileCheck %s

// llvm.func with builtin/pointer signature -> func.func; llvm.return
// inside the body becomes func.return; direct llvm.call to a raised
// callee becomes func.call.

// CHECK-LABEL: func.func private @callee
// CHECK-NOT: llvm.func @callee
llvm.func internal @callee(%a: i32, %b: i32) -> i32 {
    %0 = llvm.add %a, %b : i32
    // CHECK: return
    llvm.return %0 : i32
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: llvm.func @main
llvm.func @main(%x: i32, %y: i32) -> i32 {
    // CHECK: call @callee
    %r = llvm.call @callee(%x, %y) : (i32, i32) -> i32
    // CHECK: return
    llvm.return %r : i32
}

// Variadic functions are not raised -- keep llvm.func.
// CHECK-LABEL: llvm.func @variadic_kept
llvm.func @variadic_kept(%fmt: !llvm.ptr, ...) -> i32 attributes {} {
    %z = llvm.mlir.constant(0 : i32) : i32
    llvm.return %z : i32
}

// External declaration with no body is not raised.
// CHECK: llvm.func @external_decl
llvm.func @external_decl(!llvm.ptr) -> i32
