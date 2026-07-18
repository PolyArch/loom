// RUN: not loom-raise-opt --loom-lower-forall-to-thread \
// RUN:   --mlir-disable-threading --mlir-print-ir-after-failure \
// RUN:   --mlir-print-ir-module-scope %s 2>&1 | FileCheck %s

// Thread promotion must not infer ownership for an unmapped forall. The
// thread launch ABI cannot represent this offset, strided domain, so failure
// must preserve the complete source domain rather than treating the upper
// bound as a zero-based grid extent.

// CHECK: error: loom-lower-forall-to-thread: raw scf.forall has no recognized Loom thread mapping
// CHECK-LABEL: func.func @offset_strided(
// CHECK: scf.forall
// CHECK-SAME: (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}})
// CHECK-NOT: dataflow.thread.launch
// CHECK-NOT: dataflow.thread private

func.func @offset_strided(%buffer: memref<?xi32>) {
  %lower = arith.constant 5 : index
  %upper = arith.constant 9 : index
  %step = arith.constant 2 : index
  %value = arith.constant 7 : i32
  scf.forall (%index) = (%lower) to (%upper) step (%step) {
    memref.store %value, %buffer[%index] : memref<?xi32>
  }
  return
}
