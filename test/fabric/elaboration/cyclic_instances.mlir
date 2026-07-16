// RUN: loom --loom-elaborate-fabric-instances %s | FileCheck %s
// RUN: loom --loom-elaborate-fabric-instances %s | loom

fabric.module @identity(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.switch @WIRE [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  %result = fabric.instantiate @WIRE(
      %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %result : !fabric.bits<8>
}

// CHECK-LABEL: fabric.module @feedback
// CHECK: %[[LEFT:[^ ]+]] = fabric.switch [spatial] %[[RIGHT:[^ ]+]]
// CHECK: %[[RIGHT]] = fabric.switch [spatial]
// CHECK-SAME: %[[LEFT]]
// CHECK-NOT: fabric.instantiate
fabric.module @feedback() -> () {
  %left = fabric.instantiate @identity(
      %right : !fabric.bits<8>) -> (!fabric.bits<8>)
  %right = fabric.instantiate @identity(
      %left : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield
}
