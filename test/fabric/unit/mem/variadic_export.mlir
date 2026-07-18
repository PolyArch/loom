// RUN: loom %s | loom | FileCheck %s

// This anchor exercises the complete operation-engine ABI in one round trip:
// two manager endpoints, two subordinate endpoints, W independent of endpoint
// widths, K != P, fixed temporal slot eligibility, and export of the second
// subordinate result.

// CHECK-LABEL: fabric.module @mem_operation_engine_anchor
// CHECK: %[[MEM:.*]]:5 = fabric.mem [temporal] mgr(%{{[^,]+}}, %{{[^)]+}})
// CHECK-SAME: data_width = 32 : i32
// CHECK-SAME: dispatch_eligibility = {{\[\[}}0 : i32], [1 : i32], [0 : i32, 1 : i32]]
// CHECK-SAME: operation_table_size = 3 : i32
// CHECK: fabric.yield %[[MEM]]#1 : memref<?x!fabric.bits<16>>
fabric.module @mem_operation_engine_anchor(
    %mgr0 : memref<?x!fabric.bits<64>>,
    %mgr1 : memref<?x!fabric.bits<8>>,
    %load_addr : !fabric.bits_tag<32, 4>,
    %load_ctrl : !fabric.bits_tag<0, 4>,
    %store_addr : !fabric.bits_tag<32, 4>,
    %store_data : !fabric.bits_tag<32, 4>,
    %store_ctrl : !fabric.bits_tag<0, 4>)
    -> (memref<?x!fabric.bits<16>>) {
  %sub0, %sub1, %data, %load_done, %store_done =
      fabric.mem [temporal] mgr(%mgr0, %mgr1)
        load(%load_addr, %load_ctrl)
        store(%store_addr, %store_data, %store_ctrl)
        [{load_group_size = 1 : i32,
          store_group_size = 1 : i32,
          data_width = 32 : i32,
          tag_width = 4 : i32,
          operation_table_size = 3 : i32,
          dispatch_eligibility = [
            [0 : i32], [1 : i32], [0 : i32, 1 : i32]
          ]}]
        : (memref<?x!fabric.bits<64>>, memref<?x!fabric.bits<8>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>,
           !fabric.bits_tag<0, 4>)
        -> (memref<?x!fabric.bits<8>>, memref<?x!fabric.bits<16>>,
            !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>,
            !fabric.bits_tag<0, 4>)
  fabric.yield %sub1 : memref<?x!fabric.bits<16>>
}
