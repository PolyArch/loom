// RUN: not loom --loom-elaborate-fabric-instances --mlir-disable-threading \
// RUN:   --mlir-print-ir-after-failure -o /dev/null %s 2>&1 | FileCheck %s

fabric.module @earlier(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.switch @IDENTITY [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  %result = fabric.instantiate @IDENTITY(
      %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %result : !fabric.bits<8>
}

fabric.module @owner() -> ()
    attributes {loom_addr_bits = 48 : i32,
                loom_mem_bus_width = 256 : i32} {
  fabric.switch @OUTER [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  fabric.module @destination(%arg : !fabric.bits<8>) -> ()
      attributes {loom_addr_bits = 48 : i32,
                  loom_mem_bus_width = 512 : i32} {
    // CHECK: error: cannot materialize fabric.switch @OUTER because module-scoped semantic configuration differs
    // CHECK-SAME: loom_addr_bits definition=48 destination=48
    // CHECK-SAME: loom_mem_bus_width definition=256 destination=512
    %unused = fabric.instantiate @OUTER(
        %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
    fabric.yield
  }
  fabric.yield
}

// CHECK: IR Dump After{{.*}}ElaborateInstancesPass Failed
// CHECK-LABEL: fabric.module @earlier
// CHECK: fabric.instantiate @IDENTITY
// CHECK-LABEL: fabric.module @destination
// CHECK: fabric.instantiate @OUTER
