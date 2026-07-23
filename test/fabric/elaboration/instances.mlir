// RUN: loom --loom-elaborate-fabric-instances %s | FileCheck %s
// RUN: loom --loom-elaborate-fabric-instances %s | loom | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: loom --loom-elaborate-fabric-instances %s | not grep -q fabric.instantiate

fabric.module @leaf(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.pe @PE [spatial] (!fabric.bits<8>) -> (!fabric.bits<8>)
      attributes {sym_visibility = "private"} {
  ^bb0(%pe_arg : !fabric.bits<8>):
    fabric.fu @FU (!fabric.bits<8>) -> (!fabric.bits<8>) {
    ^bb0(%fu_arg : !fabric.bits<8>):
      %sum = fabric.op [@arith.addi] (%fu_arg, %fu_arg)
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %sum : !fabric.bits<8>
    }
    %fu_result = fabric.instantiate @FU(
        %pe_arg : !fabric.bits<8>) -> (!fabric.bits<8>)
    fabric.yield
  }
  %pe_result = fabric.instantiate @PE(
      %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %pe_result : !fabric.bits<8>
}

fabric.module @middle(%arg : !fabric.bits<16>) -> (!fabric.bits<8>) {
  %result = fabric.instantiate @leaf(
      %arg : !fabric.bits<16> to !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %result : !fabric.bits<8>
}

// CHECK-LABEL: fabric.module @top
// CHECK: %[[TOP_RESULT:.*]] = fabric.pe [spatial]
// CHECK-SAME: = %{{.*}} : !fabric.bits<32> to !fabric.bits<8>
// CHECK-NOT: sym_visibility
// CHECK: fabric.fu @FU
// CHECK: fabric.fu(
// CHECK: fabric.yield %[[TOP_RESULT]] : !fabric.bits<8>
// ROUNDTRIP-LABEL: fabric.module @top
// ROUNDTRIP: fabric.pe [spatial]
// ROUNDTRIP: fabric.fu(
fabric.module @top(%arg : !fabric.bits<32>) -> (!fabric.bits<8>) {
  %result = fabric.instantiate @middle(
      %arg : !fabric.bits<32> to !fabric.bits<16>) -> (!fabric.bits<8>)
  fabric.yield %result : !fabric.bits<8>
}

// CHECK-LABEL: fabric.module @switch_sites
// CHECK: fabric.switch @SW [spatial]
// CHECK: %[[SW0:.*]] = fabric.switch [spatial]
// CHECK: %[[SW1:.*]] = fabric.switch [spatial]
// CHECK: fabric.yield %[[SW0]], %[[SW1]]
// ROUNDTRIP-LABEL: fabric.module @switch_sites
// ROUNDTRIP: fabric.switch @SW [spatial]
// ROUNDTRIP: fabric.switch [spatial]
// ROUNDTRIP: fabric.switch [spatial]
fabric.module @switch_sites(%wide : !fabric.bits<32>,
                            %exact : !fabric.bits<8>)
    -> (!fabric.bits<8>, !fabric.bits<8>) {
  fabric.switch @SW [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  %first = fabric.instantiate @SW(
      %wide : !fabric.bits<32> to !fabric.bits<8>) -> (!fabric.bits<8>)
  %second = fabric.instantiate @SW(
      %exact : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %first, %second : !fabric.bits<8>, !fabric.bits<8>
}

// CHECK-LABEL: fabric.module @mem_site
// CHECK: fabric.mem @MEM [spatial]
// CHECK: %[[MEM_RESULT:.*]]:2 = fabric.mem [spatial]
// CHECK: fabric.yield %[[MEM_RESULT]]#0, %[[MEM_RESULT]]#1
// ROUNDTRIP-LABEL: fabric.module @mem_site
// ROUNDTRIP: fabric.mem @MEM [spatial]
// ROUNDTRIP: fabric.mem [spatial]
fabric.module @mem_site(%mgr : memref<?x!fabric.bits<32>>,
                        %addr : !fabric.bits<64>,
                        %ctrl : !fabric.bits<4>)
    -> (!fabric.bits<32>, !fabric.bits<0>) {
  fabric.mem @MEM [spatial]
      (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
  %data, %done = fabric.instantiate @MEM(
      %mgr : memref<?x!fabric.bits<32>>,
      %addr : !fabric.bits<64> to !fabric.bits<32>,
      %ctrl : !fabric.bits<4> to !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield %data, %done : !fabric.bits<32>, !fabric.bits<0>
}
