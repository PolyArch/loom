// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// The operation data width is an explicit hardware parameter.
fabric.module @mem_missing_data_width(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{'hw_params' missing required key 'data_width'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
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
        data_width = 24 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<64>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<64>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Every subordinate request source has a real manager target domain.
fabric.module @mem_empty_eligibility(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{subordinate_requests entry #0 must be non-empty}}
  %sub, %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 2 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = [[]]
        }}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (memref<?x!fabric.bits<16>>,
          !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// Strict ordering rejects duplicate manager target identities.
fabric.module @mem_duplicate_eligibility(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{operation_port_requests entry #0 must be strictly increasing}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 1 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32, 0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// Manager operand order closes the service-target identity domain.
fabric.module @mem_out_of_range_eligibility(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{operation_port_requests entry #0 manager target identity 1 is outside [0, 1)}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 1 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[1 : i32]],
          subordinate_requests = []
        }}]
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
  // expected-error @+1 {{non-canonical discardable attribute 'addr_table'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      {addr_table = []}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
fabric.module @mem_reject_mem_enable(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{non-canonical discardable attribute 'mem_enable'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      {mem_enable = true}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
fabric.module @mem_reject_memory_operation_table(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{non-canonical discardable attribute 'memory_operation_table'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      {memory_operation_table = []}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// The confirmed optional LocalMemoryService is outside this manager-backed
// operation-engine implementation slice.
fabric.module @mem_reject_local_service(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{'hw_params' key 'local_memory_service' describes the confirmed optional LocalMemoryService, which is outside this manager-backed operation-engine implementation slice}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        },
        local_memory_service = {}}]
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
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (!fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// The rejected slot-to-port array is not the manager-target relation.
fabric.module @mem_reject_slot_to_port_dispatch(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{'hw_params' key 'dispatch_eligibility' must be a dictionary}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, dispatch_eligibility = [[0 : i32]]}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Generic syntax must not bypass the canonical attribute set.
fabric.module @mem_reject_generic_discardable(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{non-canonical discardable attribute 'service_target_selection'}}
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
  }> {service_target_selection = 0 : i32}
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Every engine requires an explicit manager-target dispatch relation.
fabric.module @mem_missing_dispatch_eligibility(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{'hw_params' key 'dispatch_eligibility' must be a dictionary}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 1 : i32}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// The operation request-source cardinality is exactly P.
fabric.module @mem_dispatch_size_mismatch(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{operation_port_requests length 0 must equal physical operation port count 1}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 2 : i32,
        dispatch_eligibility = {
          operation_port_requests = [],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// Every request source has an explicit array domain.
fabric.module @mem_dispatch_non_array_row(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{operation_port_requests entry #0 must be an array}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 1 : i32,
        dispatch_eligibility = {
          operation_port_requests = [0 : i32],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// Manager target identities have one canonical signless i32 representation.
fabric.module @mem_dispatch_non_i32_identity(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 3>,
    %ctrl : !fabric.bits_tag<0, 3>) {
  // expected-error @+1 {{operation_port_requests entry #0 manager target identities must be signless i32 values}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 3 : i32,
        operation_table_size = 1 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i64]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
      -> (!fabric.bits_tag<32, 3>, !fabric.bits_tag<0, 3>)
  fabric.yield
}

// -----
// The structured relation is closed.
fabric.module @mem_dispatch_unknown_domain(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{dispatch_eligibility contains unsupported key 'slot_port_domains'}}
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = [],
          slot_port_domains = [[0 : i32]]
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// Subordinate request-source cardinality comes from leading memref results.
fabric.module @mem_subordinate_dispatch_size_mismatch(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+2 {{subordinate_requests length 0 must equal subordinate endpoint count 1}}
  %sub, %data, %done =
      fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (memref<?x!fabric.bits<16>>, !fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// One physical match domain with one tag bit represents only two rows.
fabric.module @mem_temporal_resident_capacity(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits_tag<32, 1>,
    %ctrl : !fabric.bits_tag<0, 1>) {
  // expected-error @+1 {{operation_table_size 3 exceeds representable temporal row capacity 2}}
  %data, %done = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 1 : i32,
        operation_table_size = 3 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 1>, !fabric.bits_tag<0, 1>)
      -> (!fabric.bits_tag<32, 1>, !fabric.bits_tag<0, 1>)
  fabric.yield
}

// -----
// Unknown generic properties must fail before generated decoding can drop
// them.
fabric.module @mem_reject_generic_addr_table_property(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{invalid properties}}
  %data, %done = "fabric.mem"(%mgr, %addr, %ctrl) <{
    addr_table = []
  }> : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
fabric.module @mem_reject_arbitrary_generic_property(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{unknown key}}
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
    schedule = 0 : i32,
    unowned_property = "state"
  }> : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}
