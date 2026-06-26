// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// The do-while uplift only matches the exact ne counted-loop shape and
// unsigned half-open loops. Signed less-than, inclusive predicates
// (sle / ule / sge / uge), reversed predicates (sgt / ugt, or slt / ult
// with the bumped induction value on the right-hand side), and result
// arity-changing while loops MUST NOT trigger the rewrite -- doing so
// without proving trip-count and result-shape equivalence would change
// semantics or build invalid IR.

// CHECK-LABEL: func.func @sle_predicate_kept
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @sle_predicate_kept(%buf: memref<?xf32>,
                              %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %v = memref.load %buf[%iv] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi sle, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}

// CHECK-LABEL: func.func @sgt_predicate_kept
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @sgt_predicate_kept(%buf: memref<?xf32>,
                              %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %v = memref.load %buf[%iv] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi sgt, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}

// CHECK-LABEL: func.func @slt_predicate_kept
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @slt_predicate_kept(%buf: memref<?xi32>,
                              %n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0 : i32
    %r:2 = scf.while (%iv = %c0, %acc = %z) : (index, i32) -> (index, i32) {
      %v = memref.load %buf[%iv] : memref<?xi32>
      %sum = arith.addi %acc, %v : i32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi slt, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, i32
    } do {
    ^bb0(%iv: index, %acc: i32):
      scf.yield %iv, %acc : index, i32
    }
    return %r#1 : i32
}

// CHECK-LABEL: func.func @result_arity_mismatch_kept
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @result_arity_mismatch_kept(%n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %r:3 = scf.while (%iv = %c0, %acc = %z) : (index, i32) -> (index, i32, i1) {
      %sum = arith.addi %acc, %one : i32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi ult, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum, %cond : index, i32, i1
    } do {
    ^bb0(%iv: index, %acc: i32, %flag: i1):
      scf.yield %iv, %acc : index, i32
    }
    return %r#1 : i32
}
