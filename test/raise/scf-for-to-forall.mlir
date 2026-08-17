// RUN: loom-raise-opt --loom-scf-for-to-forall %s | FileCheck %s

// A matching loop outside every callable is not owned by this pass.
// CHECK: memref.alloc
// CHECK: scf.for
// CHECK: memref.store
// CHECK-NOT: scf.forall

// Case 1: a trivially parallel scf.for with no iter_args lifts to
// scf.forall. The store inside writes to an iv-derived address, which
// is the conservative parallel-safety condition the pass requires.

// CHECK-LABEL: func.func @parallel_init
// CHECK: arith.addi
// CHECK: scf.forall
// CHECK: memref.store
// CHECK-NOT: scf.for

%module_buffer = memref.alloc() : memref<8xf32>
%module_c0 = arith.constant 0 : index
%module_c1 = arith.constant 1 : index
%module_c8 = arith.constant 8 : index
%module_f0 = arith.constant 0.0 : f32
scf.for %i = %module_c0 to %module_c8 step %module_c1 {
  memref.store %module_f0, %module_buffer[%i] : memref<8xf32>
}

func.func @parallel_init(%buf: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %unrelated = arith.addi %c0, %c1 : index
    scf.for %i = %c0 to %n step %c1 {
      memref.store %f0, %buf[%i] : memref<?xf32>
    }
    return
}

// Case 2: an scf.for with iter_args is a reduction. It must NOT be
// lifted because doing so would break the loop-carried recurrence.

// CHECK-LABEL: func.func @reduction_kept
// CHECK: scf.for {{.*}} iter_args
// CHECK-NOT: scf.forall
func.func @reduction_kept(%buf: memref<?xf32>, %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r = scf.for %i = %c0 to %n step %c1
        iter_args(%acc = %f0) -> (f32) {
      %v = memref.load %buf[%i] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      scf.yield %sum : f32
    }
    return %r : f32
}

// Case 3: nested loops. Outer is parallel (no iter_args, store at the
// end of the inner-loop body has an address depending on the outer
// iv); inner is a reduction (has iter_args). The outer must lift to
// scf.forall while the inner stays as scf.for.

// CHECK-LABEL: func.func @nested_outer_parallel
// CHECK: scf.forall
// CHECK: scf.for {{.*}} iter_args
// CHECK: memref.store
func.func @nested_outer_parallel(
    %mat: memref<?x?xf32> {llvm.noalias},
    %out: memref<?xf32> {llvm.noalias}, %m: index, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    scf.for %i = %c0 to %m step %c1 {
      %r = scf.for %j = %c0 to %n step %c1
          iter_args(%acc = %f0) -> (f32) {
        %v = memref.load %mat[%i, %j] : memref<?x?xf32>
        %s = arith.addf %acc, %v : f32
        scf.yield %s : f32
      }
      memref.store %r, %out[%i] : memref<?xf32>
    }
    return
}

// Case 4: integer-typed scf.for (i64 iv) lifts to scf.forall, with an
// arith.index_cast inserted at the top of the new body so the original
// integer-typed body operations continue to consume an i64 iv.

// CHECK-LABEL: llvm.func @parallel_i64
// CHECK: scf.forall
// CHECK: arith.index_cast {{.*}} : index to i64
// CHECK: llvm.store
// CHECK-NOT: scf.for
llvm.func @parallel_i64(%base: !llvm.ptr) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %fzero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c64 step %c1 : i64 {
      %p = llvm.getelementptr inbounds %base[%i]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      llvm.store %fzero, %p : f32, !llvm.ptr
    }
    llvm.return
}

// Lane-local structured control is still parallel when the only shared
// side effect is the iv-addressed store. This shape models kernels such
// as popcount: the outer loop distributes elements, while an inner
// scalar while loop computes each element locally.

// CHECK-LABEL: func.func @parallel_lane_local_while
// CHECK: scf.forall
// CHECK: scf.while
// CHECK: memref.store
// CHECK-LABEL: func.func @nested_parallel_loops
// CHECK-COUNT-2: scf.forall
// CHECK-NOT: scf.for %

func.func @parallel_lane_local_while(
    %input: memref<?xi32> {llvm.noalias},
    %output: memref<?xi32> {llvm.noalias}, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %i32_zero = arith.constant 0 : i32
    %i32_one = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 {
      %value = memref.load %input[%i] : memref<?xi32>
      %result:2 = scf.while (%v = %value, %count = %i32_zero) : (i32, i32) -> (i32, i32) {
        %more = arith.cmpi ne, %v, %i32_zero : i32
        scf.condition(%more) %v, %count : i32, i32
      } do {
      ^bb0(%v_next: i32, %count_next: i32):
        %bit = arith.andi %v_next, %i32_one : i32
        %updated = arith.addi %count_next, %bit : i32
        %shifted = arith.shrui %v_next, %i32_one : i32
        scf.yield %shifted, %updated : i32, i32
      }
      memref.store %result#1, %output[%i] : memref<?xi32>
    }
    return
}

func.func @nested_parallel_loops(%output: memref<?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %unused = arith.addi %j, %c1 : index
    }
    memref.store %f0, %output[%i] : memref<?xf32>
  }
  return
}
