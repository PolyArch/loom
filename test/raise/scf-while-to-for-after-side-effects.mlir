// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// A post-tested scf.while whose after-region carries a side effect is
// preserved whole: the shape is not mechanically equivalent to scf.for, so
// no rewrite fires and the store is never at risk of being dropped. The
// loop and its complete after-region stay legal scf.while.

// CHECK-LABEL: func.func @after_region_has_store
// CHECK: scf.while
// CHECK-NOT: scf.for {{.*}} iter_args
func.func @after_region_has_store(%buf: memref<?xf32>,
                                  %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.while (%iv = %c0, %acc = %f0) : (index, f32) -> (index, f32) {
      %v = memref.load %buf[%iv] : memref<?xf32>
      %sum = arith.addf %acc, %v : f32
      %iv_n = arith.addi %iv, %c1 : index
      %cond = arith.cmpi ne, %iv_n, %n : index
      scf.condition(%cond) %iv_n, %sum : index, f32
    } do {
    ^bb0(%iv: index, %acc: f32):
      // Side effect in the after-region: a memref.store. It must survive
      // the pass exactly, inside the preserved scf.while.
      memref.store %acc, %buf[%iv] : memref<?xf32>
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}
