// RUN: loom-raise-opt --loom-scf-while-to-for %s | FileCheck %s

// The do-while uplift would silently drop ops in the after-region of an
// scf.while when rewriting it into scf.for. The matcher MUST therefore
// reject after-regions that contain anything other than the
// passthrough yield (and value-preserving casts feeding it).

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
      // Side-effecting op in the after-region: a memref.store. This op
      // would be silently dropped by the do-while -> scf.for rewrite,
      // so the matcher must refuse to fire on this loop.
      memref.store %acc, %buf[%iv] : memref<?xf32>
      scf.yield %iv, %acc : index, f32
    }
    return %r#1 : f32
}
