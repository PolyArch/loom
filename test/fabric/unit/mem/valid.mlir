// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Anonymous spatial fabric.mem with both load and store ports, hw-only.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_spatial_anon_hw
fabric.module @mem_spatial_anon_hw(%mgr : memref<?x!fabric.bits<32>>,
                                   %la0 : !fabric.bits<32>, %lc0 : !fabric.bits<0>,
                                   %la1 : !fabric.bits<32>, %lc1 : !fabric.bits<0>,
                                   %sa0 : !fabric.bits<32>, %sd0 : !fabric.bits<32>,
                                   %sc0 : !fabric.bits<0>) {
  // CHECK: fabric.mem [spatial]
  // CHECK-SAME: load_group_size = 2 : i32
  // CHECK-SAME: store_group_size = 1 : i32
  %d0, %dn0, %d1, %dn1, %sdone =
      fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0, %la1, %lc1)
                            store(%sa0, %sd0, %sc0)
        [{load_group_size = 2 : i32, store_group_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>,
           !fabric.bits<32>, !fabric.bits<0>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>,
            !fabric.bits<32>, !fabric.bits<0>,
            !fabric.bits<0>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous spatial fabric.mem with memref_sub bypass result.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_spatial_with_sub
fabric.module @mem_spatial_with_sub(%mgr : memref<?x!fabric.bits<32>>,
                                    %la0 : !fabric.bits<32>, %lc0 : !fabric.bits<0>) {
  // CHECK: fabric.mem [spatial]
  // CHECK-SAME: memref<?x!fabric.bits<16>>
  %sub, %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (memref<?x!fabric.bits<16>>,
            !fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous spatial fabric.mem programmed (addr_table + mem_enable).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_spatial_anon_prog
fabric.module @mem_spatial_anon_prog(%mgr : memref<?x!fabric.bits<32>>,
                                     %la0 : !fabric.bits<32>, %lc0 : !fabric.bits<0>,
                                     %la1 : !fabric.bits<32>, %lc1 : !fabric.bits<0>,
                                     %sa0 : !fabric.bits<32>, %sd0 : !fabric.bits<32>,
                                     %sc0 : !fabric.bits<0>) {
  // CHECK: fabric.mem [spatial]
  // CHECK-SAME: addr_table
  // CHECK-SAME: mem_enable = true
  %d0, %dn0, %d1, %dn1, %sdone =
      fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0, %la1, %lc1)
                            store(%sa0, %sd0, %sc0)
        [{load_group_size = 2 : i32, store_group_size = 1 : i32}]
        {addr_table = [
            {base_addr = 65536 : i48, element_log2_size = 2 : i4, valid = true},
            {base_addr = 65792 : i48, element_log2_size = 2 : i4, valid = true},
            {base_addr = 131072 : i48, element_log2_size = 2 : i4, valid = true}
          ], mem_enable = true}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>,
           !fabric.bits<32>, !fabric.bits<0>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>,
            !fabric.bits<32>, !fabric.bits<0>,
            !fabric.bits<0>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Load-only anonymous spatial fabric.mem.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_spatial_load_only
fabric.module @mem_spatial_load_only(%mgr : memref<?x!fabric.bits<8>>,
                                     %la0 : !fabric.bits<32>, %lc0 : !fabric.bits<0>) {
  // CHECK: fabric.mem [spatial] mgr(
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<8>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<8>, !fabric.bits<0>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Store-only anonymous spatial fabric.mem.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_spatial_store_only
fabric.module @mem_spatial_store_only(%mgr : memref<?x!fabric.bits<32>>,
                                      %sa0 : !fabric.bits<32>, %sd0 : !fabric.bits<32>,
                                      %sc0 : !fabric.bits<0>) {
  // CHECK: fabric.mem [spatial] mgr(
  %sdone = fabric.mem [spatial] mgr(%mgr) store(%sa0, %sd0, %sc0)
        [{load_group_size = 0 : i32, store_group_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> !fabric.bits<0>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous temporal fabric.mem, hw-only.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_temporal_anon_hw
fabric.module @mem_temporal_anon_hw(%mgr : memref<?x!fabric.bits<32>>,
                                    %la0 : !fabric.bits_tag<32, 4>,
                                    %lc0 : !fabric.bits_tag<0, 4>) {
  // CHECK: fabric.mem [temporal]
  // CHECK-SAME: addr_table_size = 4 : i32
  // CHECK-SAME: tag_width = 4 : i32
  %d0, %dn0 = fabric.mem [temporal] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32,
          tag_width = 4 : i32, addr_table_size = 4 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous temporal fabric.mem programmed: addr_table = CAM by tag.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_temporal_anon_prog
fabric.module @mem_temporal_anon_prog(%mgr : memref<?x!fabric.bits<32>>,
                                      %la0 : !fabric.bits_tag<32, 4>,
                                      %lc0 : !fabric.bits_tag<0, 4>,
                                      %sa0 : !fabric.bits_tag<32, 4>,
                                      %sd0 : !fabric.bits_tag<32, 4>,
                                      %sc0 : !fabric.bits_tag<0, 4>) {
  // CHECK: fabric.mem [temporal]
  // CHECK-SAME: addr_table
  // CHECK-SAME: mem_enable = true
  %d0, %dn0, %sdone = fabric.mem [temporal] mgr(%mgr) load(%la0, %lc0)
                                              store(%sa0, %sd0, %sc0)
        [{load_group_size = 1 : i32, store_group_size = 1 : i32,
          tag_width = 4 : i32, addr_table_size = 2 : i32}]
        {addr_table = [
            {base_addr = 65536 : i48, element_log2_size = 2 : i4,
             tag = 3 : i4, valid = true},
            {base_addr = 131072 : i48, element_log2_size = 2 : i4,
             tag = 5 : i4, valid = true}
          ], mem_enable = true}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>,
            !fabric.bits_tag<0, 4>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Named spatial fabric.mem template (declaration only).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.mem @MyMemSpatial [spatial]
fabric.mem @MyMemSpatial [spatial]
       (memref<?x!fabric.bits<32>>,
        !fabric.bits<32>, !fabric.bits<0>,
        !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
       -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>)
       [{load_group_size = 1 : i32, store_group_size = 1 : i32}]
       {addr_table = [
           {base_addr = 0 : i48, element_log2_size = 2 : i4, valid = true},
           {base_addr = 4096 : i48, element_log2_size = 2 : i4, valid = true}
         ], mem_enable = true}

// -----------------------------------------------------------------------------
// Named temporal fabric.mem template.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.mem @MyMemTemporal [temporal]
fabric.mem @MyMemTemporal [temporal]
       (memref<?x!fabric.bits<32>>,
        !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
       -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
       [{load_group_size = 1 : i32, store_group_size = 0 : i32,
         tag_width = 4 : i32, addr_table_size = 1 : i32}]
       {addr_table = [
           {base_addr = 0 : i48, element_log2_size = 2 : i4,
            tag = 0 : i4, valid = true}
         ], mem_enable = true}

// -----------------------------------------------------------------------------
// Per-module loom_addr_bits / loom_mem_bus_width override.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_module_override
// CHECK-SAME: loom_addr_bits = 32 : i32
// CHECK-SAME: loom_mem_bus_width = 1024 : i32
fabric.module @mem_module_override(%mgr : memref<?x!fabric.bits<32>>,
                                   %la0 : !fabric.bits<32>, %lc0 : !fabric.bits<0>)
    attributes {loom_addr_bits = 32 : i32, loom_mem_bus_width = 1024 : i32} {
  // CHECK: fabric.mem [spatial]
  // CHECK-SAME: base_addr = 1024 : i32
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        {addr_table = [
            {base_addr = 1024 : i32, element_log2_size = 2 : i4, valid = true}
          ], mem_enable = true}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.mem inside module body alongside pe and fifo (whitelist sanity).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @mem_with_other_ops
fabric.module @mem_with_other_ops(%mgr : memref<?x!fabric.bits<32>>,
                                  %a : !fabric.bits<32>,
                                  %sa : !fabric.bits<32>,
                                  %sd : !fabric.bits<32>,
                                  %sc : !fabric.bits<0>) {
  // CHECK: fabric.fifo
  %f = fabric.fifo %a [max_depth = 4, bypassable = false] : !fabric.bits<32>
  // CHECK: fabric.mem [spatial]
  %sdone = fabric.mem [spatial] mgr(%mgr) store(%sa, %sd, %sc)
        [{load_group_size = 0 : i32, store_group_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> !fabric.bits<0>
  fabric.yield
}
