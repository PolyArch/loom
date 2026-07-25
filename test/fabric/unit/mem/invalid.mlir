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
// The raw hw_params dictionary is not a Local Memory Service owner; the typed
// memory_contract is the only owner.
fabric.module @mem_reject_local_service(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>) {
  // expected-error @+1 {{'hw_params' key 'local_memory_service' describes the confirmed optional LocalMemoryService, which must be represented by the typed memory_contract}}
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
  // expected-error @+1 {{operation-engine-only occurrence requires at least one manager endpoint}}
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
    memory_contract = #fabric.memory_contract<
      engine = <schedule = spatial>,
      manager_endpoints = [0],
      subordinate_endpoints = []
    >
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
// Two physical match domains with one tag bit represent only four rows.
fabric.module @mem_temporal_resident_capacity(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr0 : !fabric.bits_tag<32, 1>,
    %ctrl0 : !fabric.bits_tag<0, 1>,
    %addr1 : !fabric.bits_tag<32, 1>,
    %ctrl1 : !fabric.bits_tag<0, 1>) {
  // expected-error @+2 {{operation_table_size 5 exceeds representable temporal row capacity 4}}
  %data0, %done0, %data1, %done1 =
      fabric.mem [temporal] mgr(%mgr)
        load(%addr0, %ctrl0, %addr1, %ctrl1)
      [{load_group_size = 2 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32, tag_width = 1 : i32,
        operation_table_size = 5 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32], [0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>,
         !fabric.bits_tag<32, 1>, !fabric.bits_tag<0, 1>,
         !fabric.bits_tag<32, 1>, !fabric.bits_tag<0, 1>)
      -> (!fabric.bits_tag<32, 1>, !fabric.bits_tag<0, 1>,
          !fabric.bits_tag<32, 1>, !fabric.bits_tag<0, 1>)
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
    memory_contract = #fabric.memory_contract<
      engine = <schedule = spatial>,
      manager_endpoints = [0],
      subordinate_endpoints = []
    >,
    unowned_property = "state"
  }> : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// -----
// A fabric.mem occurrence needs an engine or a local service.
// expected-error @+1 {{requires an Operation Engine or Local Memory Service}}
"fabric.mem"() <{
  function_type = () -> (),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    manager_endpoints = [],
    subordinate_endpoints = []
  >,
  sym_name = "NoMemoryComponent"
}> : () -> ()

// -----
// A subordinate endpoint alone is not storage backing.
// expected-error @+1 {{subordinate endpoint requires an Operation Engine or Local Memory Service}}
"fabric.mem"() <{
  function_type = () -> (memref<?x!fabric.bits<32>>),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    manager_endpoints = [],
    subordinate_endpoints = [0]
  >,
  sym_name = "SubordinateWithoutOwner"
}> : () -> ()

// -----
// A manager endpoint needs an issuer even when local storage is present.
// expected-error @+1 {{manager endpoint requires an Operation Engine}}
"fabric.mem"() <{
  function_type = (memref<?x!fabric.bits<32>>) -> (),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    local_service = <
      capacity_bytes = 4096,
      service_contract = <behavior = storage>
    >,
    manager_endpoints = [0],
    subordinate_endpoints = []
  >,
  sym_name = "ManagerWithoutEngine"
}> : () -> ()

// -----
// An engine cannot be constructed without its exact schedule.
"fabric.mem"() <{
  function_type = () -> (),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    // expected-error @+3 {{expected a parameter name in struct}}
    // expected-error @+2 {{expected valid keyword}}
    // expected-error @+1 {{failed to parse Fabric_MemoryContractAttr parameter 'engine'}}
    engine = <>,
    manager_endpoints = [],
    subordinate_endpoints = []
  >,
  sym_name = "EngineWithoutSchedule"
}> : () -> ()

// -----
// Legacy schedule shorthand cannot coexist with a typed memory_contract.
// expected-error @+1 {{expected '('}}
fabric.mem @DuplicateScheduleAuthority [spatial]
    contract #fabric.memory_contract<
      engine = <schedule = temporal>,
      manager_endpoints = [0],
      subordinate_endpoints = []
    >
    (memref<?x!fabric.bits<32>>) -> ()

// -----
// Schedule is not an independently editable fabric.mem property.
// expected-error @+1 {{unknown key}}
"fabric.mem"() <{
  function_type = () -> (memref<?x!fabric.bits<32>>),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    local_service = <
      capacity_bytes = 4096,
      service_contract = <behavior = storage>
    >,
    manager_endpoints = [],
    subordinate_endpoints = [0]
  >,
  schedule = 0 : i32,
  sym_name = "ScheduleWithoutEngine"
}> : () -> ()

// -----
// Storage-only memory needs a provider-facing endpoint.
// expected-error @+1 {{storage-only occurrence requires at least one subordinate endpoint}}
"fabric.mem"() <{
  function_type = () -> (),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    local_service = <
      capacity_bytes = 4096,
      service_contract = <behavior = storage>
    >,
    manager_endpoints = [],
    subordinate_endpoints = []
  >,
  sym_name = "StorageWithoutSubordinate"
}> : () -> ()

// -----
// Storage-only memory has no owner for operation input ports.
// expected-error @+1 {{storage-only occurrence must have zero input ports}}
"fabric.mem"() <{
  function_type = (!fabric.bits<8>) -> (memref<?x!fabric.bits<32>>),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    local_service = <
      capacity_bytes = 4096,
      service_contract = <behavior = storage>
    >,
    manager_endpoints = [],
    subordinate_endpoints = [0]
  >,
  sym_name = "StorageWithStrayInput"
}> : () -> ()

// -----
// Every memref position is classified by the typed endpoint owner.
// expected-error @+1 {{memory_contract does not classify subordinate endpoint}}
"fabric.mem"() <{
  function_type = () -> (memref<?x!fabric.bits<32>>),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    local_service = <
      capacity_bytes = 4096,
      service_contract = <behavior = storage>
    >,
    manager_endpoints = [],
    subordinate_endpoints = []
  >,
  sym_name = "UnclassifiedSubordinate"
}> : () -> ()

// -----
// Local storage capacity is an explicit nonzero architecture fact.
"fabric.mem"() <{
  function_type = () -> (memref<?x!fabric.bits<32>>),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    // expected-error @+1 {{local memory service capacity_bytes must be greater than zero}}
    local_service = <
      capacity_bytes = 0,
      service_contract = <behavior = storage>
    // expected-error @+1 {{failed to parse Fabric_MemoryContractAttr parameter 'local_service'}}
    >,
    manager_endpoints = [],
    subordinate_endpoints = [0]
  >,
  sym_name = "ZeroCapacityStorage"
}> : () -> ()

// -----
// Local service behavior has no implicit or defaulted contract.
"fabric.mem"() <{
  function_type = () -> (memref<?x!fabric.bits<32>>),
  inner_input_types = [],
  memory_contract = #fabric.memory_contract<
    local_service = <
      // expected-error @+1 {{expected ','}}
      capacity_bytes = 4096
    // expected-error @+1 {{failed to parse Fabric_MemoryContractAttr parameter 'local_service'}}
    >,
    manager_endpoints = [],
    subordinate_endpoints = [0]
  >,
  sym_name = "DefaultedLocalService"
}> : () -> ()
