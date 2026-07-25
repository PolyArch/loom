// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-dfg-sim %t.dir/plain.mlir --graph plain_load --memref 0=7 --output %t.plain.json
// RUN: FileCheck %s --check-prefix=PLAIN < %t.plain.json
// RUN: loom-dfg-sim %t.dir/volatile.mlir --graph volatile_load --memref 0=7 --output %t.volatile.json
// RUN: FileCheck %s --check-prefix=VOLATILE < %t.volatile.json
// RUN: loom-dfg-sim %t.dir/atomic.mlir --graph atomic_store --memref 0=7 --output %t.atomic.json
// RUN: FileCheck %s --check-prefix=ATOMIC < %t.atomic.json
// RUN: loom-dfg-sim %t.dir/fence.mlir --graph fence_only --output %t.fence.json
// RUN: FileCheck %s --check-prefix=FENCE < %t.fence.json
// RUN: loom-dfg-sim %t.dir/rmw.mlir --graph rmw_add --memref 0=7 --output %t.rmw.json
// RUN: FileCheck %s --check-prefix=RMW < %t.rmw.json
// RUN: loom-dfg-sim %t.dir/cmpxchg.mlir --graph compare_exchange --memref 0=7 --output %t.cmpxchg.json
// RUN: FileCheck %s --check-prefix=CMPXCHG < %t.cmpxchg.json

// The plain access path is unchanged.
// PLAIN-DAG: "status": "pass"
// PLAIN-DAG: "dataflow.load": 1

// A volatile or atomic access is rejected before execution: the dynamic
// consistency-domain semantics are not implemented, and the actor must never
// be approximated as a plain access.
// VOLATILE-DAG: "event_count": 0
// VOLATILE-DAG: "status": "unsupported"
// VOLATILE-DAG: "unsupported op: dataflow.load: atomic, volatile, and fence memory contracts have no dynamic consistency-domain semantics"

// ATOMIC-DAG: "event_count": 0
// ATOMIC-DAG: "status": "unsupported"
// ATOMIC-DAG: "unsupported op: dataflow.store: atomic, volatile, and fence memory contracts have no dynamic consistency-domain semantics"

// FENCE-DAG: "event_count": 0
// FENCE-DAG: "status": "unsupported"
// FENCE-DAG: "unsupported op: dataflow.fence: atomic, volatile, and fence memory contracts have no dynamic consistency-domain semantics"

// RMW-DAG: "event_count": 0
// RMW-DAG: "status": "unsupported"
// RMW-DAG: "unsupported op: dataflow.atomic_rmw: atomic, volatile, and fence memory contracts have no dynamic consistency-domain semantics"

// CMPXCHG-DAG: "event_count": 0
// CMPXCHG-DAG: "status": "unsupported"
// CMPXCHG-DAG: "unsupported op: dataflow.cmpxchg: atomic, volatile, and fence memory contracts have no dynamic consistency-domain semantics"

//--- plain.mlir
module {
  dataflow.graph private @plain_load(%start: none, %mem: memref<1xi32>)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %data, %done = dataflow.load %mem[%addr] %start : memref<1xi32>
    dataflow.graph.return %done, %data : none, i32
  }
}

//--- volatile.mlir
module {
  dataflow.graph private @volatile_load(%start: none, %mem: memref<1xi32>)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %data, %done = dataflow.load %mem[%addr] %start
        {contract = #dataflow.plain_access<is_volatile = true>}
        : memref<1xi32>
    dataflow.graph.return %done, %data : none, i32
  }
}

//--- atomic.mlir
module {
  dataflow.graph private @atomic_store(%start: none, %mem: memref<1xi32>)
      -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 3 : i32} : i32
    %done = dataflow.store %mem[%addr] %value %start
        {contract = #dataflow.atomic_access<ordering = release,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    dataflow.graph.return values() streams() memories()
        complete(%done : none)
  }
}

//--- fence.mlir
module {
  dataflow.graph private @fence_only(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %done = dataflow.fence %start
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    dataflow.graph.return values() streams() memories()
        complete(%done : none)
  }
}

//--- rmw.mlir
module {
  dataflow.graph private @rmw_add(%start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 1 : i32} : i32
    %old, %done = dataflow.atomic_rmw %mem[%addr] %value %start
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<1xi32>
    dataflow.graph.return %done, %old : none, i32
  }
}

//--- cmpxchg.mlir
module {
  dataflow.graph private @compare_exchange(%start: none, %mem: memref<1xi32>)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %expected = dataflow.constant %start {const_value = 7 : i32} : i32
    %desired = dataflow.constant %start {const_value = 9 : i32} : i32
    %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %start
        {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                               failure_ordering = monotonic,
                                               sync_scope = <system>,
                                               source_alignment_bytes = 4>}
        : memref<1xi32> -> i1
    dataflow.graph.return %done, %old : none, i32
  }
}
