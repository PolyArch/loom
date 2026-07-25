// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// A matching loop outside every callable is not owned by this mechanical
// raising pass. Its control and side effect remain exactly where they were.
// CHECK: func.func private @observe
// CHECK: scf.while
// CHECK: func.call @observe
// CHECK: arith.addi
// CHECK: arith.cmpi ne
// CHECK: scf.condition
// CHECK: scf.yield
// CHECK-NOT: scf.for

// Counted scf.while loop with a do-while shape (the increment lives in
// the `before` region and the comparison is on the bumped iv) lifts to
// scf.for. This is the shape produced by --lift-cf-to-scf followed by
// --canonicalize on raised LLVM IR.

// CHECK-LABEL: func.func @counted_reduce_sum
// CHECK: arith.muli
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK-LABEL: llvm.func @counted_reduce_i64
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args

func.func private @observe(index)

%module_c0 = arith.constant 0 : index
%module_c1 = arith.constant 1 : index
%module_n = arith.constant 8 : index
%module_z = arith.constant 0 : i32
%module_results:2 = scf.while
    (%iv = %module_c0, %acc = %module_z) : (index, i32) -> (index, i32) {
  func.call @observe(%iv) : (index) -> ()
  %next = arith.addi %iv, %module_c1 : index
  %more = arith.cmpi ne, %next, %module_n : index
  scf.condition(%more) %next, %acc : index, i32
} do {
^bb0(%iv: index, %acc: i32):
  scf.yield %iv, %acc : index, i32
}

func.func @counted_reduce_sum(%buf: memref<?xf32>, %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %unrelated = arith.muli %c1, %c1 : index
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %v = memref.load %buf[%iv] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi ne, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}

llvm.func @counted_reduce_i64(%buf: !llvm.ptr, %n: i64) -> i32 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %z = arith.constant 0 : i32
  %r:2 = scf.while (%iv = %c0, %acc = %z) : (i64, i32) -> (i64, i32) {
    %address = llvm.getelementptr inbounds %buf[%iv]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %sum = arith.addi %acc, %value : i32
    %next = arith.addi %iv, %c1 : i64
    %more = arith.cmpi ne, %next, %n : i64
    scf.condition(%more) %next, %sum : i64, i32
  } do {
  ^bb0(%iv: i64, %acc: i32):
    scf.yield %iv, %acc : i64, i32
  }
  llvm.return %r#1 : i32
}
