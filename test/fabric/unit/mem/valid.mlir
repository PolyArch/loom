// RUN: loom %s | loom | FileCheck %s

// Spatial operation ports use an explicit data width independent of endpoint
// widths. Manager and subordinate counts come from the signature.

// CHECK-LABEL: fabric.module @mem_spatial
// CHECK: %[[MEM:.*]]:5 = fabric.mem [spatial] mgr(%{{[^,]+}}, %{{[^)]+}})
fabric.module @mem_spatial(
    %mgr0 : memref<?x!fabric.bits<64>>,
    %mgr1 : memref<?x!fabric.bits<16>>,
    %load_addr : !fabric.bits<64>,
    %load_ctrl : !fabric.bits<8>,
    %store_addr : !fabric.bits<64>,
    %store_data : !fabric.bits<16>,
    %store_ctrl : !fabric.bits<4>) {
  %sub0, %sub1, %data, %load_done, %store_done =
      fabric.mem [spatial] mgr(%mgr0, %mgr1)
        load(%load_addr, %load_ctrl)
        store(%store_addr, %store_data, %store_ctrl)
        [{load_group_size = 1 : i32,
          store_group_size = 1 : i32,
          data_width = 32 : i32,
          dispatch_eligibility = {
            operation_port_requests = [
              [0 : i32, 1 : i32], [1 : i32]
            ],
            subordinate_requests = [
              [0 : i32], [0 : i32, 1 : i32]
            ]
          }}]
        : (memref<?x!fabric.bits<64>>, memref<?x!fabric.bits<16>>,
           !fabric.bits<64> to !fabric.bits<32>,
           !fabric.bits<8> to !fabric.bits<0>,
           !fabric.bits<64> to !fabric.bits<32>,
           !fabric.bits<16> to !fabric.bits<32>,
           !fabric.bits<4> to !fabric.bits<0>)
        -> (memref<?x!fabric.bits<8>>, memref<?x!fabric.bits<32>>,
            !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>)
  fabric.yield
}

// Temporal configured-row capacity is independent of physical operation-port
// count. Port identities are load ports followed by store ports.

// CHECK-LABEL: fabric.module @mem_temporal
// CHECK: fabric.mem [temporal]
fabric.module @mem_temporal(
    %mgr : memref<?x!fabric.bits<64>>,
    %load_addr : !fabric.bits_tag<32, 4>,
    %load_ctrl : !fabric.bits_tag<0, 4>,
    %store_addr : !fabric.bits_tag<32, 4>,
    %store_data : !fabric.bits_tag<24, 4>,
    %store_ctrl : !fabric.bits_tag<0, 4>) {
  %data, %load_done, %store_done =
      fabric.mem [temporal] mgr(%mgr)
        load(%load_addr, %load_ctrl)
        store(%store_addr, %store_data, %store_ctrl)
        [{load_group_size = 1 : i32,
          store_group_size = 1 : i32,
          data_width = 24 : i32,
          tag_width = 4 : i32,
          operation_table_size = 3 : i32,
          dispatch_eligibility = {
            operation_port_requests = [[0 : i32], [0 : i32]],
            subordinate_requests = []
          }}]
        : (memref<?x!fabric.bits<64>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<24, 4>,
           !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<24, 4>, !fabric.bits_tag<0, 4>,
            !fabric.bits_tag<0, 4>)
  fabric.yield
}

// Named templates use the same signature-derived endpoint layout.

// CHECK-LABEL: fabric.mem @MemTemplate [spatial]
fabric.mem @MemTemplate [spatial]
    (memref<?x!fabric.bits<64>>, memref<?x!fabric.bits<16>>,
     !fabric.bits<32>, !fabric.bits<0>)
    -> (memref<?x!fabric.bits<8>>, memref<?x!fabric.bits<32>>,
        !fabric.bits<32>, !fabric.bits<0>)
    [{load_group_size = 1 : i32,
      store_group_size = 0 : i32,
      data_width = 32 : i32,
      dispatch_eligibility = {
        operation_port_requests = [[0 : i32, 1 : i32]],
        subordinate_requests = [[0 : i32], [1 : i32]]
      }}]

// Generic syntax accepts the same closed canonical property set and
// round-trips through the custom printer.

// CHECK-LABEL: fabric.module @mem_generic_properties
// CHECK: fabric.mem [spatial]
fabric.module @mem_generic_properties(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  %data, %done = "fabric.mem"(%mgr, %addr, %ctrl) <{
    hw_params = [{
      data_width = 32 : i32,
      dispatch_eligibility = {
        operation_port_requests = [[0 : i32]],
        subordinate_requests = []
      },
      load_group_size = 1 : i32,
      store_group_size = 0 : i32
    }],
    inner_input_types = [],
    schedule = 0 : i32
  }> : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}
