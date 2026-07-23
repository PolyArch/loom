// RUN: loom %s | loom | FileCheck %s

// A plain non-volatile access owns the canonical default contract without
// materializing a second attribute.
// CHECK-LABEL: func.func @plain_default
// CHECK: dataflow.load %{{.*}}[%{{.*}}] %{{.*}} : memref<10xi32>
// CHECK-NOT: contract
func.func @plain_default(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (i32, none) {
  %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xi32>
  return %data, %done : i32, none
}

// CHECK-LABEL: func.func @volatile_and_atomic
// CHECK: contract = #dataflow.plain_access<is_volatile = true>
// CHECK: contract = #dataflow.atomic_access<ordering = acquire, sync_scope = <system>>
// CHECK: contract = #dataflow.atomic_access<ordering = release, sync_scope = <single_thread>>
func.func @volatile_and_atomic(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (i32, i32, none) {
  %plain, %plain_done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.plain_access<is_volatile = true>}
      : memref<10xi32>
  %acquired, %acquired_done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = acquire,
                                          sync_scope = <system>>}
      : memref<10xi32>
  %stored = dataflow.store %mem[%addr] %acquired %ctrl
      {contract = #dataflow.atomic_access<ordering = release,
                                          sync_scope = <single_thread>>}
      : memref<10xi32>
  return %plain, %acquired, %stored : i32, i32, none
}

// A vector atomic access declares its granularity; a whole-payload access
// addresses one complete vector-valued memory element.
// CHECK-LABEL: func.func @vector_granularity
// CHECK: vector_granularity = per_lane
// CHECK: vector_granularity = whole_payload
func.func @vector_granularity(
    %lanes: memref<10xi32>, %payload: memref<10xvector<4xi32>>,
    %addr: index, %ctrl: none) -> (vector<4xi32>, vector<4xi32>, none, none) {
  %gathered, %gathered_done = dataflow.load %lanes[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = monotonic,
                                          sync_scope = <single_thread>,
                                          vector_granularity = per_lane>}
      : memref<10xi32>, vector<4xi32>
  %whole, %whole_done = dataflow.load %payload[%addr] %ctrl
      {contract = #dataflow.atomic_access<ordering = seq_cst,
                                          sync_scope = <system>,
                                          vector_granularity = whole_payload>}
      : memref<10xvector<4xi32>>
  return %gathered, %whole, %gathered_done, %whole_done
      : vector<4xi32>, vector<4xi32>, none, none
}

// A whole-payload compare-exchange has one atomic object, so it publishes one
// success bit even though its memory element is a vector.
// CHECK-LABEL: func.func @whole_payload_compare_exchange
// CHECK: dataflow.cmpxchg %{{.*}} : memref<8xvector<4xi32>> -> i1
func.func @whole_payload_compare_exchange(
    %mem: memref<8xvector<4xi32>>, %addr: index, %expected: vector<4xi32>,
    %desired: vector<4xi32>, %ctrl: none) -> (vector<4xi32>, i1, none) {
  %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
      {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                             failure_ordering = monotonic,
                                             sync_scope = <system>,
                                             vector_granularity = whole_payload>}
      : memref<8xvector<4xi32>> -> i1
  return %old, %ok, %done : vector<4xi32>, i1, none
}

// A multi-rank indexed access over scalar memory elements is one independent
// atomic object per lane.
// CHECK-LABEL: func.func @multi_rank_indexed_rmw
// CHECK: dataflow.atomic_rmw %{{.*}} : memref<10xi32>, vector<2x3xindex>, vector<2x3xi32>
func.func @multi_rank_indexed_rmw(
    %mem: memref<10xi32>, %addr: vector<2x3xindex>, %value: vector<2x3xi32>,
    %ctrl: none) -> (vector<2x3xi32>, none) {
  %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl
      {contract = #dataflow.rmw_contract<
          kind = add,
          access = <ordering = monotonic, sync_scope = <system>,
                    vector_granularity = per_lane>>}
      : memref<10xi32>, vector<2x3xindex>, vector<2x3xi32>
  return %old, %done : vector<2x3xi32>, none
}

// Unrelated discardable metadata is not a contract owner and stays legal.
// CHECK-LABEL: func.func @unrelated_metadata
// CHECK: dataflow.load %{{.*}} {contract = #dataflow.plain_access<is_volatile = true>, debug_label = "histogram update"}
func.func @unrelated_metadata(%mem: memref<10xi32>, %addr: index, %ctrl: none)
    -> (i32, none) {
  %data, %done = dataflow.load %mem[%addr] %ctrl
      {contract = #dataflow.plain_access<is_volatile = true>,
       debug_label = "histogram update"}
      : memref<10xi32>
  return %data, %done : i32, none
}

// The read-modify-write aggregate is exactly one kind plus one nested atomic
// access contract.
// CHECK-LABEL: func.func @read_modify_write
// CHECK: dataflow.atomic_rmw %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} {contract = #dataflow.rmw_contract<kind = fadd, access = <ordering = monotonic, sync_scope = <system>, is_volatile = true>>} : memref<10xf32>
func.func @read_modify_write(
    %mem: memref<10xf32>, %addr: index, %value: f32, %ctrl: none)
    -> (f32, none) {
  %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl
      {contract = #dataflow.rmw_contract<
          kind = fadd,
          access = <ordering = monotonic, sync_scope = <system>,
                    is_volatile = true>>}
      : memref<10xf32>
  return %old, %done : f32, none
}

// Canonical vector memory admits any positive fixed rank in row-major lane
// order; a per-lane compare-exchange publishes the exact access shape.
// CHECK-LABEL: func.func @multi_rank_per_lane
// CHECK: dataflow.cmpxchg %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} %{{.*}} mask %{{.*}} {{.*}}vector_granularity = per_lane{{.*}} : memref<10xi32>, vector<2x3xi32> -> vector<2x3xi1>
func.func @multi_rank_per_lane(
    %mem: memref<10xi32>, %addr: index, %expected: vector<2x3xi32>,
    %desired: vector<2x3xi32>, %mask: vector<2x3xi1>, %ctrl: none)
    -> (vector<2x3xi32>, vector<2x3xi1>, none) {
  %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
      mask %mask
      {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                             failure_ordering = acquire,
                                             sync_scope = <system>,
                                             vector_granularity = per_lane,
                                             weak = true>}
      : memref<10xi32>, vector<2x3xi32> -> vector<2x3xi1>
  return %old, %ok, %done : vector<2x3xi32>, vector<2x3xi1>, none
}

// CHECK-LABEL: func.func @scalar_compare_exchange
// CHECK: dataflow.cmpxchg %{{.*}}[%{{.*}}] %{{.*}} %{{.*}} %{{.*}} {contract = #dataflow.cmpxchg_contract<success_ordering = acq_rel, failure_ordering = monotonic, sync_scope = <system>>} : memref<10xi32> -> i1
func.func @scalar_compare_exchange(
    %mem: memref<10xi32>, %addr: index, %expected: i32, %desired: i32,
    %ctrl: none) -> (i32, i1, none) {
  %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
      {contract = #dataflow.cmpxchg_contract<success_ordering = acq_rel,
                                             failure_ordering = monotonic,
                                             sync_scope = <system>>}
      : memref<10xi32> -> i1
  return %old, %ok, %done : i32, i1, none
}

// CHECK-LABEL: func.func @fence
// CHECK: dataflow.fence %{{.*}} {contract = #dataflow.fence_contract<ordering = seq_cst, sync_scope = <system>>}
func.func @fence(%ctrl: none) -> none {
  %done = dataflow.fence %ctrl
      {contract = #dataflow.fence_contract<ordering = seq_cst,
                                           sync_scope = <system>>}
  return %done : none
}
