// RUN: loom-raise-opt --loom-scf-for-to-forall %s | FileCheck %s

// Loop body has a syntactic anti-dependence: every iteration writes
// buf[i] from buf[i-1]. The pass MUST NOT lift this to scf.forall;
// doing so would silently turn a serial dependence into parallel
// execution. The check is satisfied because the same base pointer
// (`%buf`) is touched by both a load and a store.

// CHECK-LABEL: func.func @anti_dependence
// CHECK: scf.for
// CHECK-NOT: scf.forall
func.func @anti_dependence(%buf: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c1 to %n step %c1 {
      %im1 = arith.subi %i, %c1 : index
      %v = memref.load %buf[%im1] : memref<?xf32>
      memref.store %v, %buf[%i] : memref<?xf32>
    }
    return
}

// True parallel-init loop with a separate base pointer for the store
// (no aliasing read). This MUST still lift to scf.forall so the
// matcher does not regress on the obviously-safe shape.

// CHECK-LABEL: func.func @disjoint_init
// CHECK: scf.forall
// CHECK-NOT: scf.for
func.func @disjoint_init(%dst: memref<?xf32>, %src: memref<?xf32>,
                         %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %src[%i] : memref<?xf32>
      memref.store %v, %dst[%i] : memref<?xf32>
    }
    return
}
