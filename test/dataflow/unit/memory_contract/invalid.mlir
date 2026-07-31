// RUN: loom %s -split-input-file -verify-diagnostics

// Pointer-addressed memory requires the exact module LLVM DataLayout. An
// index-width declaration cannot stand in for pointer representation.
module @pointer_without_layout {
  func.func @load(%service: memref<10xi32>, %pointer: !llvm.ptr,
                  %ctrl: none) -> (i32, none) {
    // expected-error @+1 {{pointer layout requires a nonempty LLVM DataLayout}}
    %data, %done = dataflow.load %service[%pointer] %ctrl
        : memref<10xi32>, !llvm.ptr
    return %data, %done : i32, none
  }
}

// -----

// Pointer-addressed atomics remain valid for ordinary scalar payloads, but a
// pointer value itself is not yet an admitted atomic object. Equal storage
// width cannot erase its provenance semantics.
module @atomic_pointer_payload attributes {
  llvm.data_layout = "e-p:64:64:64:64"
} {
  func.func @load(%mem: memref<1xi64>, %address: index, %ctrl: none)
      -> (!llvm.ptr, none) {
    // expected-error @+1 {{atomic pointer payload is unsupported}}
    %data, %done = dataflow.load %mem[%address] %ctrl
        {contract = #dataflow.atomic_access<ordering = acquire,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 8>}
        : memref<1xi64>, index, !llvm.ptr
    return %data, %done : !llvm.ptr, none
  }
}

// -----

// An atomic load rejects the publishing orderings.
func.func @atomic_load_release(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (i32, none) {
  // expected-error @+1 {{atomic load ordering must not be 'release' or 'acq_rel'}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = release,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4>}
      : memref<10xi32>
  return %data, %done : i32, none
}

// -----
// An atomic store rejects the consuming orderings.
func.func @atomic_store_acquire(
    %mem: memref<10xi32>, %addr: index, %data: i32, %ctrl: none) -> none {
  // expected-error @+1 {{atomic store ordering must not be 'acquire' or 'acq_rel'}}
  %done = dataflow.store %mem[%addr] %data %ctrl
      {contract = #dataflow.atomic_access<ordering = acquire,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4>}
      : memref<10xi32>
  return %done : none
}

// -----
// An atomic read-modify-write is at least monotonic.
func.func @rmw_unordered(
    %mem: memref<10xi32>, %addr: index, %value: i32, %ctrl: none)
    -> (i32, none) {
  // expected-error @+1 {{atomic read-modify-write ordering must not be 'unordered'}}
  %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl
      {contract = #dataflow.rmw_contract<
          kind = add,
          access = <ordering = unordered, sync_scope = <system>,
                    source_alignment_bytes = 4>>}
      : memref<10xi32>
  return %old, %done : i32, none
}

// -----
// A compare-exchange failure ordering never publishes.
func.func @cmpxchg_failure_acq_rel(
    %mem: memref<10xi32>, %addr: index, %expected: i32, %desired: i32,
    %ctrl: none) -> (i32, i1, none) {
  %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
      // expected-error @+1 {{compare-exchange failure ordering must not be 'release' or 'acq_rel'}}
      {contract = #dataflow.cmpxchg_contract<success_ordering = acq_rel,
                                             failure_ordering = acq_rel,
                                             sync_scope = <system>,
                                             source_alignment_bytes = 4>}
      : memref<10xi32> -> i1
  return %old, %ok, %done : i32, i1, none
}

// -----
// A fence carries no monotonic or unordered ordering.
func.func @fence_monotonic(%ctrl: none) -> none {
  %done = dataflow.fence %ctrl
      // expected-error @+1 {{fence ordering must be 'acquire', 'release', 'acq_rel', or 'seq_cst'}}
      {contract = #dataflow.fence_contract<ordering = monotonic,
                                           sync_scope = <system>>}
  return %done : none
}

// -----
// A target scope is representable, but no authority can prove the key yet, so
// an actor referencing one is rejected.
func.func @unresolved_target_scope(%ctrl: none) -> none {
  // expected-error @+1 {{target synchronization scope 'nvptx::block' is unresolved}}
  %done = dataflow.fence %ctrl
      {contract = #dataflow.fence_contract<
          ordering = seq_cst, sync_scope = <target, "nvptx", "block">>}
  return %done : none
}

// -----
// The closed sum's target arm always names a target-namespaced key.
func.func @unnamed_target_scope(%ctrl: none) -> none {
  // expected-error @+2 {{'target' synchronization scope requires a target namespace and key}}
  // expected-error @+1 {{failed to parse}}
  %done = dataflow.fence %ctrl {contract = #dataflow.fence_contract<ordering = seq_cst, sync_scope = <target>>}
  return %done : none
}

// -----
// A scalar atomic access has one atomic object and declares no granularity.
func.func @scalar_granularity(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (i32, none) {
  // expected-error @+1 {{scalar atomic access must not declare a vector atomic granularity}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = monotonic,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4,
                                          vector_granularity = per_lane>}
      : memref<10xi32>
  return %data, %done : i32, none
}

// -----
// A vector atomic access must declare which objects are atomic.
func.func @missing_granularity(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (vector<4xi32>, none) {
  // expected-error @+1 {{vector atomic access must declare a vector atomic granularity}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = monotonic,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4>}
      : memref<10xi32>, vector<4xi32>
  return %data, %done : vector<4xi32>, none
}

// -----
// Whole-payload granularity needs one complete memory element.
func.func @whole_payload_contiguous(
    %mem: memref<10xi32>, %addr: index, %ctrl: none) -> (vector<4xi32>, none) {
  // expected-error @+1 {{'whole_payload' atomic granularity requires an access to one complete memory element}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = monotonic,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4,
                                          vector_granularity = whole_payload>}
      : memref<10xi32>, vector<4xi32>
  return %data, %done : vector<4xi32>, none
}

// -----
// One complete vector-valued memory element is one atomic object, not a lane
// group, so its element access rejects per-lane granularity.
func.func @per_lane_vector_element(
    %mem: memref<10xvector<4xi32>>, %addr: index, %ctrl: none)
    -> (vector<4xi32>, none) {
  // expected-error @+1 {{'per_lane' atomic granularity requires an access to independent memory elements}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = monotonic,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4,
                                          vector_granularity = per_lane>}
      : memref<10xvector<4xi32>>
  return %data, %done : vector<4xi32>, none
}

// -----
// A whole-payload compare-exchange has one atomic object, so its success
// result stays scalar even when the memory element is a vector.
func.func @whole_payload_success_shape(
    %mem: memref<10xvector<4xi32>>, %addr: index, %expected: vector<4xi32>,
    %desired: vector<4xi32>, %ctrl: none)
    -> (vector<4xi32>, vector<4xi1>, none) {
  // expected-error @+1 {{scalar or 'whole_payload' compare-exchange success result must be 'i1'}}
  %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
      {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                             failure_ordering = monotonic,
                                             sync_scope = <system>,
                                             source_alignment_bytes = 4,
                                             vector_granularity = whole_payload>}
      : memref<10xvector<4xi32>> -> vector<4xi1>
  return %old, %ok, %done : vector<4xi32>, vector<4xi1>, none
}

// -----
// A per-lane compare-exchange publishes one success bit per access lane.
func.func @per_lane_success_shape(
    %mem: memref<10xi32>, %addr: index, %expected: vector<4xi32>,
    %desired: vector<4xi32>, %ctrl: none) -> (vector<4xi32>, i1, none) {
  // expected-error @+1 {{'per_lane' compare-exchange success result must be 'vector<4xi1>'}}
  %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
      {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                             failure_ordering = monotonic,
                                             sync_scope = <system>,
                                             source_alignment_bytes = 4,
                                             vector_granularity = per_lane>}
      : memref<10xi32>, vector<4xi32> -> i1
  return %old, %ok, %done : vector<4xi32>, i1, none
}

// -----
// A read-modify-write action applies to its exact scalar element category.
func.func @rmw_float_action_on_integer(
    %mem: memref<10xi32>, %addr: index, %value: i32, %ctrl: none)
    -> (i32, none) {
  // expected-error @+1 {{atomicrmw 'fadd' operand must have floating-point element type, got 'i32'}}
  %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl
      {contract = #dataflow.rmw_contract<
          kind = fadd,
          access = <ordering = monotonic, sync_scope = <system>,
                    source_alignment_bytes = 4>>}
      : memref<10xi32>
  return %old, %done : i32, none
}

// -----
// An atomic object has a power-of-two byte-sized width.
func.func @rmw_non_power_of_two(
    %mem: memref<10xi7>, %addr: index, %value: i7, %ctrl: none)
    -> (i7, none) {
  // expected-error @+1 {{atomic object 'i7' size 7 must be a power of two of at least 8 bits}}
  %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl
      {contract = #dataflow.rmw_contract<
          kind = add,
          access = <ordering = monotonic, sync_scope = <system>,
                    source_alignment_bytes = 4>>}
      : memref<10xi7>
  return %old, %done : i7, none
}

// -----
// A compare-exchange compares an exact bit pattern, so it rejects a
// floating-point element.
func.func @cmpxchg_float_element(
    %mem: memref<10xf32>, %addr: index, %expected: f32, %desired: f32,
    %ctrl: none) -> (f32, i1, none) {
  // expected-error @+1 {{compare-exchange operand must have integer element type, got 'f32'}}
  %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
      {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                             failure_ordering = monotonic,
                                             sync_scope = <system>,
                                             source_alignment_bytes = 4>}
      : memref<10xf32> -> i1
  return %old, %ok, %done : f32, i1, none
}

// -----
// An atomic width that is not an exact static fact fails closed.
func.func @atomic_index_element(
    %mem: memref<10xindex>, %addr: index, %ctrl: none) -> (index, none) {
  // expected-error @+1 {{atomic load operand must have integer or floating-point element type, got 'index'}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = acquire,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4>}
      : memref<10xindex>
  return %data, %done : index, none
}

// -----
// One actor owns exactly one aggregate memory contract.
func.func @second_contract(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (i32, none) {
  // expected-error @+1 {{'shadow' must not carry a second aggregate memory contract}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.plain_access<is_volatile = true>,
       shadow = #dataflow.atomic_access<ordering = monotonic,
                                        sync_scope = <system>,
                                        source_alignment_bytes = 4>}
      : memref<10xi32>
  return %data, %done : i32, none
}

// -----
// The named contract slot also owns the synchronization scope; no alternate
// attribute name may state a second one.
func.func @second_sync_scope(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (i32, none) {
  // expected-error @+1 {{'scope' must not carry a second synchronization scope}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = acquire,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 4>,
       scope = #dataflow.sync_scope<single_thread>}
      : memref<10xi32>
  return %data, %done : i32, none
}

// -----
// Source alignment is identity-critical typed state and must be a nonzero
// power of two; it is never inferred from type or service.
func.func @atomic_load_zero_alignment(%mem: memref<10xi32>, %addr: index,
                                      %ctrl: none) -> (i32, none) {
  // expected-error @+2 {{source alignment must be a nonzero power of two}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = acquire,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 0>}
      : memref<10xi32>
  return %data, %done : i32, none
}

// -----
func.func @atomic_load_non_power_of_two_alignment(
    %mem: memref<10xi32>, %addr: index, %ctrl: none) -> (i32, none) {
  // expected-error @+2 {{source alignment must be a nonzero power of two}}
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = acquire,
                                          sync_scope = <system>,
                                          source_alignment_bytes = 3>}
      : memref<10xi32>
  return %data, %done : i32, none
}
