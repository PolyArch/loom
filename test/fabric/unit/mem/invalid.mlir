// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// The operation data width is an explicit hardware parameter.
fabric.module @mem_missing_data_width(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{'hw_params' missing required key 'data_width'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Operation data ports use W, not a manager endpoint width.
fabric.module @mem_data_width_mismatch(
    %mgr : memref<?x!fabric.bits<64>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{load data port #0 must have operation data width 24}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 24 : i32}]
      : (memref<?x!fabric.bits<64>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<64>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Temporal hardware must give every configured slot an eligibility domain.
fabric.module @mem_empty_eligibility(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{dispatch_eligibility entry #1 must be non-empty}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 2 : i32,
        dispatch_eligibility = [[0 : i32], []]}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// Strict ordering rejects duplicate physical port identities.
fabric.module @mem_duplicate_eligibility(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{dispatch_eligibility entry #0 must be strictly increasing}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 1 : i32,
        dispatch_eligibility = [[0 : i32, 0 : i32]]}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// P = L + S closes the physical port identity domain.
fabric.module @mem_out_of_range_eligibility(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{dispatch_eligibility entry #0 port identity 1 is outside [0, 1)}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 1 : i32,
        dispatch_eligibility = [[1 : i32]]}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// Workload configuration does not belong to canonical Fabric.
fabric.module @mem_reject_addr_table(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{does not accept workload configuration 'addr_table'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32}]
      {addr_table = []}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
fabric.module @mem_reject_mem_enable(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{does not accept workload configuration 'mem_enable'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32}]
      {mem_enable = true}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
fabric.module @mem_reject_memory_operation_table(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{does not accept workload configuration 'memory_operation_table'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32}]
      {memory_operation_table = []}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Local service remains unsupported without its confirmed typed contract.
fabric.module @mem_reject_local_service(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{'hw_params' contains unsupported key 'local_memory_service'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, local_memory_service = {}}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Engine-only memory requires a manager backing path.
fabric.module @mem_no_manager(
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{operation engine requires at least one manager endpoint}}
  %data, %done = fabric.mem [spatial] mgr() load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32}]
      : (!fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Spatial engines have no configured-slot capacity or eligibility table.
fabric.module @mem_spatial_dispatch(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{spatial fabric.mem must not carry temporal-only key 'dispatch_eligibility'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, dispatch_eligibility = [[0 : i32]]}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}
