// RUN: not loom --loom-elaborate-fabric-instances --mlir-disable-threading \
// RUN:   --mlir-print-ir-after-failure -o /dev/null %s 2>&1 | FileCheck %s

fabric.module @identity(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.yield %arg : !fabric.bits<8>
}

fabric.module @feedback() -> () {
  // CHECK: error: cannot eliminate fabric.module instance feedback cycle with no physical producer
  %left = fabric.instantiate @identity(
      %right : !fabric.bits<8>) -> (!fabric.bits<8>)
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  %right = fabric.instantiate @identity(
      %left : !fabric.bits<8>) -> (!fabric.bits<8>)
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  fabric.yield
}

// CHECK: IR Dump After{{.*}}ElaborateInstancesPass Failed
// CHECK-LABEL: fabric.module @feedback
// CHECK: %[[LEFT:[^ ]+]] = fabric.instantiate @identity(%[[RIGHT:[^ ]+]]
// CHECK: %[[RIGHT]] = fabric.instantiate @identity(%[[LEFT]]
