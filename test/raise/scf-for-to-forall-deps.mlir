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

// Different SSA roots are not an alias proof. Without explicit provenance the
// source and destination may denote the same object, so this sibling remains
// serial.

// CHECK-LABEL: func.func @unproven_distinct_init
// CHECK: scf.for
// CHECK-NOT: scf.forall
func.func @unproven_distinct_init(%dst: memref<?xf32>, %src: memref<?xf32>,
                                  %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %src[%i] : memref<?xf32>
      memref.store %v, %dst[%i] : memref<?xf32>
    }
    return
}

// Explicit noalias provenance on both roots proves the same elementwise
// transfer independent and permits the parallel candidate.

// CHECK-LABEL: func.func @disjoint_init
// CHECK: scf.forall
// CHECK-NOT: scf.for
func.func @disjoint_init(%dst: memref<?xf32> {llvm.noalias},
                         %src: memref<?xf32> {llvm.noalias},
                         %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %src[%i] : memref<?xf32>
      memref.store %v, %dst[%i] : memref<?xf32>
    }
    return
}

// In-place elementwise loops are safe when every access to the shared
// base uses the same iv-derived address. The read and write are ordered
// within one iteration, while different iterations touch different
// elements.

// CHECK-LABEL: func.func @inplace_same_element_update
// CHECK: scf.forall
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: scf.for
func.func @inplace_same_element_update(%buf: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %scale = arith.constant 2.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %buf[%i] : memref<?xf32>
      %scaled = arith.mulf %v, %scale : f32
      memref.store %scaled, %buf[%i] : memref<?xf32>
    }
    return
}

// Same-base read/write loops are not safe when the addresses differ:
// reading buf[i + 1] and writing buf[i] has a cross-iteration overlap.

// CHECK-LABEL: func.func @same_base_read_write_shifted
// CHECK: scf.for
// CHECK-NOT: scf.forall
func.func @same_base_read_write_shifted(%buf: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %next = arith.addi %i, %c1 : index
      %v = memref.load %buf[%next] : memref<?xf32>
      memref.store %v, %buf[%i] : memref<?xf32>
    }
    return
}

// Same-base writes are still parallel when every iteration writes a
// fixed-width lane group and the lane offsets are distinct modulo the
// per-iteration stride.

// CHECK-LABEL: func.func @disjoint_lane_stores_same_base
// CHECK: scf.forall
// CHECK-COUNT-3: memref.store
// CHECK-NOT: scf.for
func.func @disjoint_lane_stores_same_base(
    %dst: memref<?xf32> {llvm.noalias},
    %src: memref<?xf32> {llvm.noalias}, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %n step %c1 {
      %base = arith.muli %i, %c3 : index
      %lane1 = arith.addi %base, %c1 : index
      %lane2 = arith.addi %base, %c2 : index
      %a = memref.load %src[%base] : memref<?xf32>
      %b = memref.load %src[%lane1] : memref<?xf32>
      %c = memref.load %src[%lane2] : memref<?xf32>
      memref.store %a, %dst[%base] : memref<?xf32>
      memref.store %b, %dst[%lane1] : memref<?xf32>
      memref.store %c, %dst[%lane2] : memref<?xf32>
    }
    return
}

// Adjacent stores with stride one may alias across iterations
// (`i + 1` in one iteration is `i` in the next), so they must not lift.

// CHECK-LABEL: func.func @overlapping_lane_stores_same_base
// CHECK: scf.for
// CHECK-NOT: scf.forall
func.func @overlapping_lane_stores_same_base(%dst: memref<?xf32>,
                                             %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    scf.for %i = %c0 to %n step %c1 {
      %next = arith.addi %i, %c1 : index
      memref.store %f0, %dst[%i] : memref<?xf32>
      memref.store %f1, %dst[%next] : memref<?xf32>
    }
    return
}
