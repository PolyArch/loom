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
// RUN: loom-dfg-sim %t.dir/fence-hb.mlir --graph fence_hb --memref 0=0 --memref 1=0 --output %t.fence-hb.json
// RUN: FileCheck %s --check-prefix=FENCE-HB < %t.fence-hb.json
// RUN: loom-dfg-sim %t.dir/vector.mlir --graph vector_atomic --memref 0=1,2 --output %t.vector.json
// RUN: FileCheck %s --check-prefix=VECTOR < %t.vector.json
// RUN: not loom-dfg-sim %t.dir/target.mlir --graph target_scope --memref 0=1 --output %t.target.json 2>&1 | FileCheck %s --check-prefix=TARGET

// The plain access path is unchanged.
// PLAIN-DAG: "status": "pass"
// PLAIN-DAG: "dataflow.load": 1

// Volatile actions retain every dynamic observation. The two stores cannot be
// merged and the final load cannot be forwarded from a discarded operation.
// VOLATILE: "final_memory_state": {
// VOLATILE: "i32:5"
// VOLATILE: "final_outputs": [
// VOLATILE-NEXT: "none",
// VOLATILE-NEXT: "i32:5"
// VOLATILE: "dataflow.load": 1
// VOLATILE: "dataflow.store": 2
// VOLATILE: "status": "pass"

// ATOMIC: "final_memory_state": {
// ATOMIC: "i32:3"
// ATOMIC: "dataflow.store": 1
// ATOMIC: "status": "pass"

// FENCE: "dataflow.fence": 1
// FENCE: "status": "pass"

// RMW: "final_memory_state": {
// RMW: "i32:8"
// RMW: "final_outputs": [
// RMW-NEXT: "none",
// RMW-NEXT: "i32:7"
// RMW: "dataflow.atomic_rmw": 1
// RMW: "status": "pass"

// A strong success, comparison failure, and weak success prove that only
// successful exchanges append writes, old/success/done retire together, and
// this exact model makes its declared no-spurious-failure weak choice.
// CMPXCHG: "final_memory_state": {
// CMPXCHG: "i32:13"
// CMPXCHG: "final_outputs": [
// CMPXCHG-NEXT: "none",
// CMPXCHG-NEXT: "i32:7",
// CMPXCHG-NEXT: "i1:true",
// CMPXCHG-NEXT: "i32:9",
// CMPXCHG-NEXT: "i1:false",
// CMPXCHG-NEXT: "i32:9",
// CMPXCHG-NEXT: "i1:true"
// CMPXCHG: "dataflow.cmpxchg": 3
// CMPXCHG: "status": "pass"

// The release/acquire fence chain is the only relation ordering the two plain
// data accesses. Dropping either fence role makes the run unsupported on the
// unordered conflict rather than still producing 42.
// FENCE-HB: "final_outputs": [
// FENCE-HB-NEXT: "none",
// FENCE-HB-NEXT: "i32:42"
// FENCE-HB: "dataflow.fence": 2
// FENCE-HB: "status": "pass"

// Valid contracts outside this provider profile fail before execution rather
// than becoming scalar or system-scope operations.
// VECTOR-DAG: "event_count": 0
// VECTOR-DAG: "status": "unsupported"
// VECTOR-DAG: DFG-sim atomic and volatile memory provider supports scalar addressed actions only
// TARGET: target synchronization scope 'gpu::cta' is unresolved

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
    %three = dataflow.constant %start {const_value = 3 : i32} : i32
    %five = dataflow.constant %start {const_value = 5 : i32} : i32
    %first = dataflow.store %mem[%addr] %three %start
        {contract = #dataflow.plain_access<is_volatile = true>}
        : memref<1xi32>
    %second = dataflow.store %mem[%addr] %five %first
        {contract = #dataflow.plain_access<is_volatile = true>}
        : memref<1xi32>
    %data, %done = dataflow.load %mem[%addr] %second
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
                                            sync_scope = <single_thread>,
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
      -> (i32, i1, i32, i1, i32, i1)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 6, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %expected = dataflow.constant %start {const_value = 7 : i32} : i32
    %desired = dataflow.constant %start {const_value = 9 : i32} : i32
    %failed_desired = dataflow.constant %start {const_value = 11 : i32} : i32
    %weak_desired = dataflow.constant %start {const_value = 13 : i32} : i32
    %old0, %ok0, %done0 = dataflow.cmpxchg %mem[%addr] %expected %desired %start
        {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                               failure_ordering = monotonic,
                                               sync_scope = <system>,
                                               source_alignment_bytes = 4>}
        : memref<1xi32> -> i1
    %old1, %ok1, %done1 = dataflow.cmpxchg
        %mem[%addr] %expected %failed_desired %done0
        {contract = #dataflow.cmpxchg_contract<success_ordering = acq_rel,
                                               failure_ordering = acquire,
                                               sync_scope = <system>,
                                               source_alignment_bytes = 4>}
        : memref<1xi32> -> i1
    %old2, %ok2, %done2 = dataflow.cmpxchg
        %mem[%addr] %desired %weak_desired %done1
        {contract = #dataflow.cmpxchg_contract<success_ordering = acq_rel,
                                               failure_ordering = acquire,
                                               sync_scope = <system>,
                                               source_alignment_bytes = 4,
                                               weak = true>}
        : memref<1xi32> -> i1
    dataflow.graph.return values(%old0, %ok0, %old1, %ok1, %old2, %ok2
        : i32, i1, i32, i1, i32, i1) streams() memories()
        complete(%done2 : none)
  }
}

//--- fence-hb.mlir
module {
  dataflow.graph private @fence_hb(
      %start: none, %data: memref<1xi32>, %flag: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 42 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %data_done = dataflow.store %data[%zero] %value %start : memref<1xi32>
    %release = dataflow.fence %data_done
        {contract = #dataflow.fence_contract<ordering = release,
                                             sync_scope = <system>>}
    %flag_done = dataflow.store %flag[%zero] %one %release
        {contract = #dataflow.atomic_access<ordering = monotonic,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>

    // The delay makes the provider select the published flag version without
    // introducing a memory-order edge between the producer and consumer.
    %delay0:2 = dataflow.sync %start, %zero
        : (none, index) -> (none, index)
    %delay1:2 = dataflow.sync %start, %delay0#1
        : (none, index) -> (none, index)
    %delay2:2 = dataflow.sync %start, %delay1#1
        : (none, index) -> (none, index)
    %observed, %flag_read = dataflow.load %flag[%delay2#1] %start
        {contract = #dataflow.atomic_access<ordering = monotonic,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    %acquire = dataflow.fence %flag_read
        {contract = #dataflow.fence_contract<ordering = acquire,
                                             sync_scope = <system>>}
    %loaded, %done = dataflow.load %data[%zero] %acquire : memref<1xi32>
    dataflow.graph.return values(%loaded : i32) streams() memories()
        complete(%done, %flag_done : none, none)
  }
}

//--- vector.mlir
module {
  dataflow.graph private @vector_atomic(
      %start: none, %mem: memref<2xi32>) -> (vector<2xi32>)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start
        {const_value = dense<[1, 1]> : vector<2xi32>} : vector<2xi32>
    %old, %done = dataflow.atomic_rmw %mem[%addr] %value %start
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4,
                      vector_granularity = per_lane>>}
        : memref<2xi32>, vector<2xi32>
    dataflow.graph.return values(%old : vector<2xi32>) streams() memories()
        complete(%done : none)
  }
}

//--- target.mlir
module {
  dataflow.graph private @target_scope(%start: none, %mem: memref<1xi32>)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value, %done = dataflow.load %mem[%addr] %start
        {contract = #dataflow.atomic_access<ordering = acquire,
                                            sync_scope = <target, "gpu", "cta">,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
}
