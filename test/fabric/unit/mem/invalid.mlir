// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Bad schedule keyword.
fabric.module @mem_bad_schedule(%mgr : memref<?x!fabric.bits<32>>,
                                %la0 : !fabric.bits<32>, %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{expected fabric mem schedule keyword 'spatial' or 'temporal', got 'bogus'}}
  %d0, %dn0 = fabric.mem [bogus] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// load_group_size + store_group_size both 0.
fabric.module @mem_zero_groups(%mgr : memref<?x!fabric.bits<32>>) {
  // expected-error @+1 {{load_group_size + store_group_size must be >= 1}}
  fabric.mem [spatial] mgr(%mgr)
        [{load_group_size = 0 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>) -> ()
  fabric.yield
}

// -----
// tag_width present in spatial.
fabric.module @mem_spatial_tag_width(%mgr : memref<?x!fabric.bits<32>>,
                                     %la0 : !fabric.bits<32>,
                                     %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{spatial fabric.mem must not carry temporal-only attribute 'tag_width'}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32,
          tag_width = 4 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// addr_table_size missing in temporal mode.
fabric.module @mem_temporal_no_addr_table_size(%mgr : memref<?x!fabric.bits<32>>,
                                               %la0 : !fabric.bits_tag<32, 4>,
                                               %lc0 : !fabric.bits_tag<0, 4>) {
  // expected-error @+1 {{'hw_params' missing required key 'addr_table_size'}}
  %d0, %dn0 = fabric.mem [temporal] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32,
          tag_width = 4 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  fabric.yield
}

// -----
// tag_width missing in temporal mode.
fabric.module @mem_temporal_no_tag_width(%mgr : memref<?x!fabric.bits<32>>,
                                         %la0 : !fabric.bits_tag<32, 4>,
                                         %lc0 : !fabric.bits_tag<0, 4>) {
  // expected-error @+1 {{'hw_params' missing required key 'tag_width'}}
  %d0, %dn0 = fabric.mem [temporal] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32,
          addr_table_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  fabric.yield
}

// -----
// memref_mgr element type not bits<W>: i32 element type rejected.
fabric.module @mem_bad_mgr_elem(%mgr : memref<?xi32>,
                                %la0 : !fabric.bits<32>,
                                %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{memref_mgr element type must be '!fabric.bits<W>'}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?xi32>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// data port width != memref_mgr element width on a store port.
fabric.module @mem_store_data_width_mismatch(%mgr : memref<?x!fabric.bits<32>>,
                                             %sa0 : !fabric.bits<32>,
                                             %sd0 : !fabric.bits<16>,
                                             %sc0 : !fabric.bits<0>) {
  // expected-error @+1 {{store data port width mismatch with memref_mgr element width}}
  %sdone = fabric.mem [spatial] mgr(%mgr) store(%sa0, %sd0, %sc0)
        [{load_group_size = 0 : i32, store_group_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<16>, !fabric.bits<0>)
        -> !fabric.bits<0>
  fabric.yield
}

// -----
// addr port width != index_width (default 32).
fabric.module @mem_addr_width_mismatch(%mgr : memref<?x!fabric.bits<32>>,
                                       %la0 : !fabric.bits<16>,
                                       %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{schedule mismatch with port kind}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<16>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Incoming endpoint typing may normalize widths but cannot change kind.
fabric.module @mem_input_kind_mismatch(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>,
    %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{must share the same fabric kind (bits or bits_tag)}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32,
          tag_width = 4 : i32, addr_table_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32> to !fabric.bits_tag<32, 4>,
           !fabric.bits<0> to !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  fabric.yield
}

// -----
// Memory capabilities cannot use transport width-normalization syntax.
fabric.module @mem_manager_to_type(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>,
    %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{memref capabilities cannot use the 'to <destination-type>' clause}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>> to memref<?x!fabric.bits<16>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// A same-name discardable attribute must not shadow a valid inherent property.
fabric.module @mem_inner_input_types_collision(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>,
    %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{discardable attribute 'inner_input_types' conflicts with the inherent property of the same name}}
  %data, %done = "fabric.mem"(%mgr, %addr, %ctrl) <{
    hw_params = [{load_group_size = 1 : i32, store_group_size = 0 : i32}],
    inner_input_types = [memref<?x!fabric.bits<32>>, !fabric.bits<32>,
                         !fabric.bits<0>],
    schedule = 0 : i32
  }> {inner_input_types = "not-an-array"}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
     -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Schedule + port type-kind mismatch (spatial schedule with bits_tag ports).
fabric.module @mem_spatial_with_tag(%mgr : memref<?x!fabric.bits<32>>,
                                    %la0 : !fabric.bits_tag<32, 4>,
                                    %lc0 : !fabric.bits_tag<0, 4>) {
  // expected-error @+1 {{schedule mismatch with port kind}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  fabric.yield
}

// -----
// addr_table length mismatch (spatial: should equal load + store).
fabric.module @mem_addr_table_len(%mgr : memref<?x!fabric.bits<32>>,
                                  %la0 : !fabric.bits<32>,
                                  %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{'addr_table' length 2 must equal load_group_size + store_group_size (1)}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        {addr_table = [
            {base_addr = 0 : i48, element_log2_size = 2 : i4, valid = true},
            {base_addr = 4096 : i48, element_log2_size = 2 : i4, valid = true}
          ], mem_enable = true}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// element_log2_size out of range (default loom_mem_bus_width = 32768 -> log2(4096) = 12).
fabric.module @mem_log2_too_big(%mgr : memref<?x!fabric.bits<32>>,
                                %la0 : !fabric.bits<32>,
                                %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{'element_log2_size' value 13 exceeds log2(loom_mem_bus_width / 8) = 12}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        {addr_table = [
            {base_addr = 0 : i48, element_log2_size = 13 : i4, valid = true}
          ], mem_enable = true}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Temporal duplicate valid tag.
fabric.module @mem_temporal_dup_tag(%mgr : memref<?x!fabric.bits<32>>,
                                    %la0 : !fabric.bits_tag<32, 4>,
                                    %lc0 : !fabric.bits_tag<0, 4>) {
  // expected-error @+1 {{temporal duplicate valid tag value 3}}
  %d0, %dn0 = fabric.mem [temporal] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32,
          tag_width = 4 : i32, addr_table_size = 2 : i32}]
        {addr_table = [
            {base_addr = 0 : i48, element_log2_size = 2 : i4,
             tag = 3 : i4, valid = true},
            {base_addr = 4096 : i48, element_log2_size = 2 : i4,
             tag = 3 : i4, valid = true}
          ], mem_enable = true}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  fabric.yield
}

// -----
// All-or-nothing violation: addr_table present but mem_enable missing.
fabric.module @mem_only_addr_table(%mgr : memref<?x!fabric.bits<32>>,
                                   %la0 : !fabric.bits<32>,
                                   %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{all-or-nothing violation: 'addr_table' is present but 'mem_enable' is missing}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        {addr_table = [
            {base_addr = 0 : i48, element_log2_size = 2 : i4, valid = true}
          ]}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// All-or-nothing violation: mem_enable present but addr_table missing.
fabric.module @mem_only_enable(%mgr : memref<?x!fabric.bits<32>>,
                               %la0 : !fabric.bits<32>,
                               %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{all-or-nothing violation: 'mem_enable' is present but 'addr_table' is missing}}
  %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        {mem_enable = true}
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// memref_sub element type not bits<W_sub>: i32 element type rejected.
fabric.module @mem_bad_sub_elem(%mgr : memref<?x!fabric.bits<32>>,
                                %la0 : !fabric.bits<32>,
                                %lc0 : !fabric.bits<0>) {
  // expected-error @+1 {{memref_sub element type must be '!fabric.bits<W_sub>'}}
  %sub, %d0, %dn0 = fabric.mem [spatial] mgr(%mgr) load(%la0, %lc0)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (memref<?xi32>,
            !fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}
