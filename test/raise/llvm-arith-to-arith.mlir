// RUN: loom-raise-opt --loom-llvm-func-to-func --loom-llvm-arith-to-arith %s | FileCheck %s

// Verify that LLVM arithmetic, compare and constant ops with builtin
// numeric types are rewritten into the matching arith dialect ops.
// Pointer-typed ops (gep/load/store/alloca) and llvm.* float-ext / cast
// ops with exotic types stay in the llvm dialect on purpose. The
// arith-to-arith pass is nested under func.func, so the inputs are
// first raised via loom-llvm-func-to-func.

// CHECK-LABEL: func.func @int_arith
llvm.func @int_arith(%a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.addi %arg0, %arg1
    %0 = llvm.add %a, %b : i32
    // CHECK: %{{.*}} = arith.muli
    %1 = llvm.mul %0, %a : i32
    // CHECK: %{{.*}} = arith.subi
    %2 = llvm.sub %1, %b : i32
    llvm.return %2 : i32
}

// CHECK-LABEL: func.func @float_arith
llvm.func @float_arith(%a: f32, %b: f32) -> f32 {
    // CHECK: %{{.*}} = arith.addf %arg0, %arg1
    %0 = llvm.fadd %a, %b : f32
    // CHECK: %{{.*}} = arith.mulf
    %1 = llvm.fmul %0, %a : f32
    llvm.return %1 : f32
}

// CHECK-LABEL: func.func @int_cmp
llvm.func @int_cmp(%a: i32, %b: i32) -> i1 {
    // CHECK: %{{.*}} = arith.cmpi slt, %arg0, %arg1
    %0 = llvm.icmp "slt" %a, %b : i32
    llvm.return %0 : i1
}

// CHECK-LABEL: func.func @float_cmp
llvm.func @float_cmp(%a: f32, %b: f32) -> i1 {
    // CHECK: %{{.*}} = arith.cmpf oeq, %arg0, %arg1
    %0 = llvm.fcmp "oeq" %a, %b : f32
    llvm.return %0 : i1
}

// CHECK-LABEL: func.func @numeric_select
llvm.func @numeric_select(%cond: i1, %a: i32, %b: i32) -> i32 {
    // CHECK: %{{.*}} = arith.select %arg0, %arg1, %arg2 : i32
    %0 = llvm.select %cond, %a, %b : i1, i32
    llvm.return %0 : i32
}

// CHECK-LABEL: func.func @int_constant
llvm.func @int_constant() -> i32 {
    // CHECK: %{{.*}} = arith.constant 42 : i32
    %0 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %0 : i32
}

// CHECK-LABEL: func.func @ptr_load_stays_llvm
// Load of a pointer-typed memory remains as llvm.load -- we only
// rewrite arith and comparison ops here.
llvm.func @ptr_load_stays_llvm(%p: !llvm.ptr) -> f32 {
    // CHECK: %{{.*}} = llvm.load
    %v = llvm.load %p : !llvm.ptr -> f32
    llvm.return %v : f32
}

// CHECK-LABEL: func.func @ptr_select_stays_llvm
llvm.func @ptr_select_stays_llvm(%cond: i1, %a: !llvm.ptr, %b: !llvm.ptr) -> !llvm.ptr {
    // CHECK: %{{.*}} = llvm.select
    %0 = llvm.select %cond, %a, %b : i1, !llvm.ptr
    llvm.return %0 : !llvm.ptr
}
