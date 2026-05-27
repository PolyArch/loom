// RUN: loom-raise-opt --loom-llvm-func-to-func --loom-llvm-cf-to-cf --loom-llvm-arith-to-arith %s | FileCheck %s

// A builtin-signature caller `@fn1` invokes an aggregate-signature
// callee `@fn2`. After the pipeline, `@fn1` is raised to func.func,
// while `@fn2` STAYS as `llvm.func` because of its struct argument.
// `@fn1` retains its `llvm.call @fn2` -- this is the documented mixed-
// dialect island shape that downstream layers must handle.

// CHECK-LABEL: llvm.func @fn2
// CHECK-SAME: !llvm.struct
// CHECK: llvm.return
llvm.func @fn2(%s: !llvm.struct<(i32, i32)>) -> i32 {
    %z = llvm.mlir.constant(0 : i32) : i32
    llvm.return %z : i32
}

// CHECK-LABEL: func.func @fn1
// CHECK-NOT: llvm.func @fn1
// CHECK: llvm.call @fn2
// CHECK: return
llvm.func @fn1() -> i32 {
    %z = llvm.mlir.constant(0 : i32) : i32
    %u = llvm.mlir.undef : !llvm.struct<(i32, i32)>
    %r = llvm.call @fn2(%u) : (!llvm.struct<(i32, i32)>) -> i32
    llvm.return %r : i32
}
