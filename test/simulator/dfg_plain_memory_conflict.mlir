// RUN: loom-dfg-sim %s --graph unordered_ww_overlap --arg 0=8589934593 \
// RUN:   --memref 1=10,11,12,13,14,15,16,17 --output %t.ww.json
// RUN: FileCheck %s --check-prefix=WW < %t.ww.json
// RUN: loom-dfg-sim %s --graph ordered_rw_chain \
// RUN:   --memref 0=10,11,12,13,14,15,16,17 --output %t.chain.json
// RUN: FileCheck %s --check-prefix=CHAIN < %t.chain.json
// RUN: loom-dfg-sim %s --graph unordered_repeated_store \
// RUN:   --arg 0=2 --arg 1=0 --arg 2=-1 --arg 3=7 \
// RUN:   --memref 4=10,11,12,13,14,15,16,17 --output %t.repeat.json
// RUN: FileCheck %s --check-prefix=REPEAT < %t.repeat.json
// RUN: loom-dfg-sim %s --graph ordered_repeated_store \
// RUN:   --arg 0=2 --arg 1=0 --arg 2=-1 --arg 3=7 \
// RUN:   --memref 4=10,11,12,13,14,15,16,17 --output %t.repeat-ok.json
// RUN: FileCheck %s --check-prefix=REPEAT-OK < %t.repeat-ok.json
// RUN: loom-dfg-sim %s --graph unordered_plain_then_atomic \
// RUN:   --memref 0=7 --output %t.plain-atomic.json
// RUN: FileCheck %s --check-prefix=PLAIN-ATOMIC < %t.plain-atomic.json
// RUN: loom-dfg-sim %s --graph unordered_atomic_then_plain \
// RUN:   --memref 0=7 --output %t.atomic-plain.json
// RUN: FileCheck %s --check-prefix=ATOMIC-PLAIN < %t.atomic-plain.json
// RUN: loom-dfg-sim %s --graph unordered_atomic_pair \
// RUN:   --memref 0=0 --output %t.atomic-pair.json
// RUN: FileCheck %s --check-prefix=ATOMIC-PAIR < %t.atomic-pair.json
// RUN: loom-dfg-sim %s --graph unordered_atomic_pair_reversed \
// RUN:   --memref 0=0 --output %t.atomic-pair-reversed.json
// RUN: FileCheck %s --check-prefix=ATOMIC-PAIR < %t.atomic-pair-reversed.json
// RUN: loom-dfg-sim %s --graph ordered_plain_then_atomic \
// RUN:   --memref 0=7 --output %t.ordered-plain-atomic.json
// RUN: FileCheck %s --check-prefix=ORDERED-PLAIN-ATOMIC < %t.ordered-plain-atomic.json
// RUN: loom-dfg-sim %s --graph ordered_atomic_then_plain \
// RUN:   --memref 0=7 --output %t.ordered-atomic-plain.json
// RUN: FileCheck %s --check-prefix=ORDERED-ATOMIC-PLAIN < %t.ordered-atomic-plain.json
// RUN: loom-dfg-sim %s --graph ordered_plain_read_then_failed_cmp \
// RUN:   --memref 0=7 --output %t.plain-read-failed-cmp.json
// RUN: FileCheck %s --check-prefix=PLAIN-READ-FAILED-CMP < %t.plain-read-failed-cmp.json

// The Memory section of docs/spec-sim-dfg.md requires that plain conflicting
// accesses without an explicit causal order never become deterministic through
// simulator traversal order: such a run reports an unsupported capability
// instead of an arbitrary result or a deadlock witness. An explicit done/ctrl
// chain is the one thing that orders them.

// Two unordered plain stores overlap on one element: the contiguous vector
// store covers elements 1 and 2 and the scalar store covers element 2. The
// scheduler must reject the complete ready set before either store fires, so
// the run exports no output, memory state, or memory-root result.
// WW-DAG: "final_memory_roots": {}
// WW-DAG: "final_memory_state": {}
// WW-DAG: "final_outputs": []
// WW-NOT: "dataflow.store":
// WW: "status": "unsupported"

// The same element under an explicit done/ctrl chain is ordered, so both
// accesses fire and the load observes the stored value.
// CHAIN: "final_outputs": [
// CHAIN-NEXT: "none",
// CHAIN-NEXT: "i32:42"
// CHAIN: "dataflow.load": 1
// CHAIN: "dataflow.store": 1
// CHAIN: "status": "pass"

// One store actor writes a fixed cell on every loop iteration. Successive
// firings of one static actor are unordered unless an explicit token frontier
// links them; sharing the same operation is not an order. The first firing
// commits and the second, meeting the unordered conflict, does not fire.
// REPEAT: "dataflow.store": 1
// REPEAT: "status": "unsupported"

// The same repeats become legal when a memory-frontier carry threads each
// store's done into the next firing's ctrl, so every firing commits.
// REPEAT-OK: "dataflow.store": 2
// REPEAT-OK: "status": "pass"

// Mixed plain/atomic hazards are still data races, independent of which actor
// has the lower structural ordinal. Admission rejects the whole ready set
// before candidate visitation can turn either case into a plain fallback.
// PLAIN-ATOMIC-NOT: "dataflow.load":
// PLAIN-ATOMIC-NOT: "dataflow.store":
// PLAIN-ATOMIC: unordered plain accesses conflict on the same memory
// PLAIN-ATOMIC: "status": "unsupported"
// ATOMIC-PLAIN-NOT: "dataflow.load":
// ATOMIC-PLAIN-NOT: "dataflow.store":
// ATOMIC-PLAIN: unordered plain accesses conflict on the same memory
// ATOMIC-PLAIN: "status": "unsupported"

// Atomic/atomic overlap is legal. MemoryAtomicOrder serializes the two RMWs,
// and canonical ActorRef order selects one deterministic allowed execution
// while both operations retain their atomic old-value semantics.
// ATOMIC-PAIR: "final_memory_state": {
// ATOMIC-PAIR: "i32:3"
// ATOMIC-PAIR: "final_outputs": [
// ATOMIC-PAIR-NEXT: "none",
// ATOMIC-PAIR-NEXT: "i32:0",
// ATOMIC-PAIR-NEXT: "i32:1"
// ATOMIC-PAIR: "dataflow.atomic_rmw": 2
// ATOMIC-PAIR: "status": "pass"

// A causal edge removes the data race but cannot make a plain write a member
// of atomic modification order. Until the exact model owns a value/version
// correspondence for mixed storage, both directions fail closed after the
// first access instead of reporting a false reads-from relation.
// ORDERED-PLAIN-ATOMIC: mixed atomic/plain write hazard has no exact DFG value/version correspondence
// ORDERED-PLAIN-ATOMIC: "dataflow.store": 1
// ORDERED-PLAIN-ATOMIC-NOT: "dataflow.load":
// ORDERED-PLAIN-ATOMIC: "status": "unsupported"
// ORDERED-ATOMIC-PLAIN: mixed atomic/plain write hazard has no exact DFG value/version correspondence
// ORDERED-ATOMIC-PLAIN: "dataflow.store": 1
// ORDERED-ATOMIC-PLAIN-NOT: "dataflow.load":
// ORDERED-ATOMIC-PLAIN: "status": "unsupported"

// A failed compare-exchange is a read. Once its comparison is known, an
// earlier ordered plain read is not misclassified as a mixed write hazard.
// PLAIN-READ-FAILED-CMP: "final_memory_state": {
// PLAIN-READ-FAILED-CMP: "i32:7"
// PLAIN-READ-FAILED-CMP: "final_outputs": [
// PLAIN-READ-FAILED-CMP-NEXT: "none",
// PLAIN-READ-FAILED-CMP-NEXT: "i32:7",
// PLAIN-READ-FAILED-CMP-NEXT: "i32:7",
// PLAIN-READ-FAILED-CMP-NEXT: "i1:false"
// PLAIN-READ-FAILED-CMP: "dataflow.cmpxchg": 1
// PLAIN-READ-FAILED-CMP: "dataflow.load": 1
// PLAIN-READ-FAILED-CMP: "status": "pass"

module {
  dataflow.graph private @unordered_ww_overlap(
      %start: none, %packed: i64, %mem: memref<8xi32>) -> ()
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data = dataflow.unpack %packed : i64 -> vector<2xi32>
    %base = dataflow.constant %start {const_value = 1 : index} : index
    %cell = dataflow.constant %start {const_value = 2 : index} : index
    %value = dataflow.constant %start {const_value = 200 : i32} : i32
    %vector_done = dataflow.store %mem[%base] %data %start
        : memref<8xi32>, vector<2xi32>
    %scalar_done = dataflow.store %mem[%cell] %value %start : memref<8xi32>
    dataflow.graph.return values() streams() memories()
        complete(%vector_done, %scalar_done : none, none)
  }

  dataflow.graph private @ordered_rw_chain(
      %start: none, %mem: memref<8xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %cell = dataflow.constant %start {const_value = 4 : index} : index
    %value = dataflow.constant %start {const_value = 42 : i32} : i32
    %store_done = dataflow.store %mem[%cell] %value %start : memref<8xi32>
    %data, %load_done = dataflow.load %mem[%cell] %store_done : memref<8xi32>
    dataflow.graph.return values(%data : i32) streams() memories()
        complete(%load_done : none)
  }

  dataflow.graph private @unordered_repeated_store(
      %ctrl: none, %ub: i16, %lb: i16, %step: i16, %value: i8,
      %mem: memref<8xi8>) -> ()
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %index, %rwc = dataflow.stream %ub, %lb, %step step add while sgt : i16
    %execution = dataflow.carry %rwc, %ctrl, %execution_lane#1 : none
    %execution_lane:2 = dataflow.demux %rwc, %execution
        : (i1, none) -> (none, none)
    %memory_frontier = dataflow.carry %rwc, %ctrl, %store_done : none
    %memory_lane:2 = dataflow.demux %rwc, %memory_frontier
        : (i1, none) -> (none, none)
    %stable_value = dataflow.invariant %rwc, %value : i8
    %zero_i16 = dataflow.constant %ctrl {const_value = 0 : i16} : i16
    %stable_index = dataflow.invariant %rwc, %zero_i16 : i16
    %idx = arith.index_cast %stable_index : i16 to index
    // The store's ctrl is the execution lane, never its own prior done, so no
    // token frontier orders the repeats.
    %store_done = dataflow.store %mem[%idx] %stable_value %execution_lane#1
        : memref<8xi8>
    %retired:2 = dataflow.sync %execution_lane#0, %memory_lane#0
        : (none, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%retired#0 : none)
  }

  dataflow.graph private @ordered_repeated_store(
      %ctrl: none, %ub: i16, %lb: i16, %step: i16, %value: i8,
      %mem: memref<8xi8>) -> ()
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %index, %rwc = dataflow.stream %ub, %lb, %step step add while sgt : i16
    %execution = dataflow.carry %rwc, %ctrl, %execution_lane#1 : none
    %execution_lane:2 = dataflow.demux %rwc, %execution
        : (i1, none) -> (none, none)
    %memory_frontier = dataflow.carry %rwc, %ctrl, %store_done : none
    %memory_lane:2 = dataflow.demux %rwc, %memory_frontier
        : (i1, none) -> (none, none)
    %stable_value = dataflow.invariant %rwc, %value : i8
    %zero_i16 = dataflow.constant %ctrl {const_value = 0 : i16} : i16
    %stable_index = dataflow.invariant %rwc, %zero_i16 : i16
    %idx = arith.index_cast %stable_index : i16 to index
    // The store's ctrl is the memory frontier that threads its own prior done,
    // so an explicit none-token chain orders the repeats.
    %store_done = dataflow.store %mem[%idx] %stable_value %memory_lane#1
        : memref<8xi8>
    %retired:2 = dataflow.sync %execution_lane#0, %memory_lane#0
        : (none, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%retired#0 : none)
  }

  dataflow.graph private @unordered_plain_then_atomic(
      %start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 9 : i32} : i32
    %plain_done = dataflow.store %mem[%addr] %value %start : memref<1xi32>
    %observed, %atomic_done = dataflow.load %mem[%addr] %start
        {contract = #dataflow.atomic_access<ordering = monotonic,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    dataflow.graph.return values(%observed : i32) streams() memories()
        complete(%plain_done, %atomic_done : none, none)
  }

  dataflow.graph private @unordered_atomic_then_plain(
      %start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 9 : i32} : i32
    %atomic_done = dataflow.store %mem[%addr] %value %start
        {contract = #dataflow.atomic_access<ordering = monotonic,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    %observed, %plain_done = dataflow.load %mem[%addr] %start : memref<1xi32>
    dataflow.graph.return values(%observed : i32) streams() memories()
        complete(%atomic_done, %plain_done : none, none)
  }

  dataflow.graph private @unordered_atomic_pair(
      %start: none, %mem: memref<1xi32>) -> (i32, i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %two = dataflow.constant %start {const_value = 2 : i32} : i32
    %old0, %done0 = dataflow.atomic_rmw %mem[%addr] %one %start
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<1xi32>
    %old1, %done1 = dataflow.atomic_rmw %mem[%addr] %two %start
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<1xi32>
    dataflow.graph.return values(%old0, %old1 : i32, i32) streams() memories()
        complete(%done0, %done1 : none, none)
  }

  // Source presentation is deliberately reversed. Canonical ActorRef order,
  // not this operation order, must select the same exact-model execution.
  dataflow.graph private @unordered_atomic_pair_reversed(
      %start: none, %mem: memref<1xi32>) -> (i32, i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %two = dataflow.constant %start {const_value = 2 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %old1, %done1 = dataflow.atomic_rmw %mem[%addr] %two %start
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<1xi32>
    %old0, %done0 = dataflow.atomic_rmw %mem[%addr] %one %start
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<1xi32>
    dataflow.graph.return values(%old0, %old1 : i32, i32) streams() memories()
        complete(%done0, %done1 : none, none)
  }

  dataflow.graph private @ordered_plain_then_atomic(
      %start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 9 : i32} : i32
    %plain_done = dataflow.store %mem[%addr] %value %start : memref<1xi32>
    %observed, %atomic_done = dataflow.load %mem[%addr] %plain_done
        {contract = #dataflow.atomic_access<ordering = monotonic,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    dataflow.graph.return values(%observed : i32) streams() memories()
        complete(%atomic_done : none)
  }

  dataflow.graph private @ordered_atomic_then_plain(
      %start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 9 : i32} : i32
    %atomic_done = dataflow.store %mem[%addr] %value %start
        {contract = #dataflow.atomic_access<ordering = monotonic,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    %observed, %plain_done = dataflow.load %mem[%addr] %atomic_done
        : memref<1xi32>
    dataflow.graph.return values(%observed : i32) streams() memories()
        complete(%plain_done : none)
  }

  dataflow.graph private @ordered_plain_read_then_failed_cmp(
      %start: none, %mem: memref<1xi32>) -> (i32, i32, i1)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 3, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %expected = dataflow.constant %start {const_value = 8 : i32} : i32
    %desired = dataflow.constant %start {const_value = 9 : i32} : i32
    %plain, %plain_done = dataflow.load %mem[%addr] %start : memref<1xi32>
    %old, %ok, %cmp_done = dataflow.cmpxchg
        %mem[%addr] %expected %desired %plain_done
        {contract = #dataflow.cmpxchg_contract<
            success_ordering = acq_rel, failure_ordering = acquire,
            sync_scope = <system>, source_alignment_bytes = 4>}
        : memref<1xi32> -> i1
    dataflow.graph.return values(%plain, %old, %ok : i32, i32, i1)
        streams() memories() complete(%cmp_done : none)
  }
}
