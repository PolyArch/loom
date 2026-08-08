// RUN: not loom --loom-elaborate-fabric-instances --mlir-disable-threading \
// RUN:   --mlir-print-ir-after-failure -o /dev/null %s 2>&1 | FileCheck %s

fabric.module @callee() -> ()
    attributes {loom_addr_bits = 32 : i32,
                loom_mem_bus_width = 256 : i32} {
  fabric.yield
}

fabric.module @earlier(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.switch @IDENTITY [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  %result = fabric.instantiate @IDENTITY(
      %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %result : !fabric.bits<8>
}

fabric.module @later() -> ()
    attributes {loom_addr_bits = 48 : i32,
                loom_mem_bus_width = 512 : i32} {
  // CHECK: error: cannot inline fabric.module @callee because module-scoped semantic configuration differs
  // CHECK-SAME: loom_addr_bits callee=32 caller=48
  // CHECK-SAME: loom_mem_bus_width callee=256 caller=512
  fabric.instantiate @callee() -> ()
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  fabric.yield
}

// CHECK: IR Dump After{{.*}}ElaborateInstancesPass Failed
// CHECK-LABEL: fabric.module @earlier
// CHECK: fabric.instantiate @IDENTITY
// CHECK-LABEL: fabric.module @later
// CHECK: fabric.instantiate @callee
