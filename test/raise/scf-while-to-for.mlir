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

// The counted do-while shape that CFG-to-SCF structuring emits (increment
// and ne/ult comparison in the `before` region, condition on the bumped
// induction value) is not mechanically equivalent to scf.for: its body runs
// at least once even when %lb >= %ub, it can fail to
// terminate, and the failed condition forwards a value that need not equal
// the scf.for exit value. Proving equivalence needs a loop-semantics analysis
// the pass does not own, so these loops stay as legal scf.while. An unrelated
// arith op in the same callable is left untouched.

// CHECK-LABEL: func.func @counted_reduce_sum
// CHECK: arith.muli
// CHECK: scf.while
// CHECK: arith.cmpi ne
// CHECK: scf.condition
// CHECK-NOT: scf.for
// CHECK-LABEL: llvm.func @counted_reduce_i64
// CHECK: scf.while
// CHECK: arith.cmpi ne
// CHECK: scf.condition
// CHECK-NOT: scf.for

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

// The pre-tested counted shape -- the `before` block holds only the slt
// exit comparison against a loop-invariant bound and the induction bump
// lives in the `after` block -- has the standard scf.for trip count, so
// the upstream utility uplifts it. Both loop results are dead here, so the
// exit induction value the utility reconstructs from the trip count
// (instead of forwarding the exact failed-condition value) is unobservable
// and the rewrite is exact. The body effect and the imported loop
// annotation move to the scf.for; the reconstruction arithmetic the
// utility emits for the exit value stays dead behind it.

// CHECK-LABEL: func.func @pre_tested_counted_uplifts
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}})
// CHECK: memref.store %[[ACC]], %{{.*}}[%[[IV]]] : memref<?xf32>
// CHECK: scf.yield %[[ACC]] : f32
// CHECK: } {llvm.loop_annotation = #loop_annotation}
// CHECK-NOT: scf.while

#loop_annotation = #llvm.loop_annotation<mustProgress = true>

func.func @pre_tested_counted_uplifts(%buf: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %cond = arith.cmpi slt, %iv, %n : index
      scf.condition(%cond) %iv, %acc : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      memref.store %acc, %buf[%iv] : memref<?xf32>
      %iv_n = arith.addi %iv, %c1 : index
      scf.yield %iv_n, %acc : index, f32
    } attributes {llvm.loop_annotation = #loop_annotation}
    return
}

// Same pre-tested counted shape, but a loop result escapes: the failed
// scf.condition forwards the exact exit induction value, which the utility
// would replace with its trip-count reconstruction, so the loop must stay
// scf.while. Removing the pass's no-external-users gate fails this anchor.
// The dead-result sibling in the same callable still uplifts: candidacy is
// decided per loop, not per callable.

// CHECK-LABEL: func.func @observable_result_kept
// CHECK: %[[KEPT:.*]] = scf.while
// CHECK: arith.cmpi slt
// CHECK: scf.condition
// CHECK: scf.for
// CHECK: return %[[KEPT]] : index
func.func @observable_result_kept(%n: index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %kept = scf.while (%iv = %c0) : (index) -> index {
      %cond = arith.cmpi slt, %iv, %n : index
      scf.condition(%cond) %iv : index
    } do {
    ^bb0(%iv: index):
      %iv_n = arith.addi %iv, %c3 : index
      scf.yield %iv_n : index
    }
    %dead:2 = scf.while (%iv = %c0, %j = %c0) : (index, index) -> (index, index) {
      %cond = arith.cmpi slt, %iv, %n : index
      scf.condition(%cond) %iv, %j : index, index
    } do {
    ^bb0(%iv: index, %j: index):
      %iv_n = arith.addi %iv, %c1 : index
      %j_n = arith.addi %j, %c3 : index
      scf.yield %iv_n, %j_n : index, index
    }
    return %kept : index
}
