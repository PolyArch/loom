// RUN: loom-dfg-sim %s --graph unordered_ww_overlap --arg 0=8589934593 \
// RUN:   --memref 1=10,11,12,13,14,15,16,17 --output %t.ww.json
// RUN: FileCheck %s --check-prefix=WW < %t.ww.json
// RUN: loom-dfg-sim %s --graph unordered_ww_overlap_reversed \
// RUN:   --arg 0=8589934593 --memref 1=10,11,12,13,14,15,16,17 \
// RUN:   --output %t.ww-reversed.json
// RUN: FileCheck %s --check-prefix=WW < %t.ww-reversed.json
// RUN: loom-dfg-sim %s --graph unordered_ww_overlap --arg 0=8589934593 \
// RUN:   --memref 1=10,11,12,13,14,15,16,17 --max-event-steps 2 \
// RUN:   --output %t.limit.json
// RUN: FileCheck %s --check-prefix=LIMIT < %t.limit.json
// RUN: loom-dfg-sim %s --graph unordered_rw_overlap \
// RUN:   --arg 0=0x0000000100000003 --memref 1=10,11,12,13,14,15,16,17 \
// RUN:   --output %t.rw.json
// RUN: FileCheck %s --check-prefix=RW < %t.rw.json
// RUN: loom-dfg-sim %s --graph uninitialized_unordered_rw_precedence \
// RUN:   --output %t.uninitialized-rw.json
// RUN: FileCheck %s --check-prefix=UNINITIALIZED-RW \
// RUN:   < %t.uninitialized-rw.json
// RUN: loom-dfg-sim %s --graph conflict_precedes_projection_error \
// RUN:   --memref 0=10 --output %t.projection-error.json
// RUN: FileCheck %s --check-prefix=PROJECTION-ERROR \
// RUN:   < %t.projection-error.json
// RUN: loom-dfg-sim %s --graph ordered_rw_chain \
// RUN:   --memref 0=10,11,12,13,14,15,16,17 --output %t.chain.json
// RUN: FileCheck %s --check-prefix=CHAIN < %t.chain.json
// RUN: loom-dfg-sim %s --graph data_dependency_does_not_order_memory \
// RUN:   --memref 0=17 --output %t.data-dependency.json
// RUN: FileCheck %s --check-prefix=DATA-DEPENDENCY \
// RUN:   < %t.data-dependency.json
// RUN: loom-dfg-sim %s --graph empty_store_does_not_launder_order \
// RUN:   --memref 0=17 --output %t.empty-store.json
// RUN: FileCheck %s --check-prefix=EMPTY-STORE < %t.empty-store.json
// RUN: loom-dfg-sim %s --graph unordered_rr_overlap \
// RUN:   --memref 0=10,11,12,13,14,15,16,17 --output %t.rr.json
// RUN: FileCheck %s --check-prefix=RR < %t.rr.json
// RUN: loom-dfg-sim %s --graph unordered_ww_disjoint \
// RUN:   --memref 0=10,11,12,13,14,15,16,17 --output %t.disjoint.json
// RUN: FileCheck %s --check-prefix=DISJOINT < %t.disjoint.json
// RUN: loom-dfg-sim %s --graph masked_inactive_lane_no_access \
// RUN:   --arg 0=425201762392 --arg 1=0x0000000500000002 --arg 2=2 \
// RUN:   --memref 3=10,11,12,13,14,15,16,17 --output %t.masked.json
// RUN: FileCheck %s --check-prefix=MASKED < %t.masked.json
// RUN: loom-dfg-sim %s --graph parallelize_group_inherits_lane_order \
// RUN:   --memref 0=10,11,12,13,14,15,16,17 --output %t.group.json
// RUN: FileCheck %s --check-prefix=GROUP < %t.group.json
// RUN: loom-dfg-sim %s --graph unordered_repeated_store \
// RUN:   --arg 0=2 --arg 1=0 --arg 2=-1 --arg 3=7 \
// RUN:   --memref 4=10,11,12,13,14,15,16,17 --output %t.repeat.json
// RUN: FileCheck %s --check-prefix=REPEAT < %t.repeat.json
// RUN: loom-dfg-sim %s --graph ordered_repeated_store \
// RUN:   --arg 0=2 --arg 1=0 --arg 2=-1 --arg 3=7 \
// RUN:   --memref 4=10,11,12,13,14,15,16,17 --output %t.repeat-ok.json
// RUN: FileCheck %s --check-prefix=REPEAT-OK < %t.repeat-ok.json

// The Memory section of docs/spec-sim-dfg.md requires that plain conflicting
// accesses without an explicit causal order never become deterministic
// through simulator traversal order: such a run reports an unsupported
// capability instead of an arbitrary result or a deadlock witness. Reads do
// not conflict with reads, disjoint byte ranges do not conflict, an explicit
// done/ctrl chain orders an overlap, and an inactive mask lane derives no
// access at all (Dynamic Memory Action Projection in
// docs/spec-dataflow-memory-consistency.md).

// Two unordered plain stores overlap on one element: the contiguous vector
// store covers elements 1 and 2 and the scalar store covers element 2. The
// scheduler must reject the complete ready set before either store fires,
// independently of their textual order. Unsupported execution exports no
// output, memory state, or memory-root result.
// WW: "final_memory_roots": {}
// WW-NEXT: "final_memory_state": {}
// WW-NEXT: "final_outputs": []
// WW-NOT: "dataflow.store":
// WW: "status": "unsupported"

// Reaching the event limit after discovering the same conflict must preserve
// the formal unsupported result rather than relabel it as blocked, and it must
// not append the event-limit diagnostic that a genuine block would carry.
// LIMIT: "status": "unsupported"
// LIMIT-NOT: "maximum event steps reached"

// An unordered plain scalar store to element 3 overlaps lane 0 of an
// unordered indexed load of elements 3 and 1. Which access meets the conflict
// is again not an observable, so only the terminal status is asserted.
// RW: "status": "unsupported"

// A load from fresh storage would be uninitialized in isolation, but its
// unordered overlap with a ready store is the scheduler decision that takes
// precedence. The complete access set must be projected before either access
// executes, so the symmetric conflict reports unsupported without an
// uninitialized-read diagnostic or terminal state.
// UNINITIALIZED-RW: "diagnostics": [
// UNINITIALIZED-RW-NOT: uninitialized
// UNINITIALIZED-RW: "final_memory_roots": {}
// UNINITIALIZED-RW-NEXT: "final_memory_state": {}
// UNINITIALIZED-RW-NEXT: "final_outputs": []
// UNINITIALIZED-RW-NOT: "dataflow.load":
// UNINITIALIZED-RW-NOT: "dataflow.store":
// UNINITIALIZED-RW: "status": "unsupported"

// A ready out-of-range access has a projection diagnostic, but it cannot mask
// the knowable conflict between two other ready stores. Admission projects the
// complete ready set locally and gives the symmetric conflict precedence,
// without exporting terminal state.
// PROJECTION-ERROR: "diagnostics": [
// PROJECTION-ERROR-NOT: out of range
// PROJECTION-ERROR: "final_memory_roots": {}
// PROJECTION-ERROR-NEXT: "final_memory_state": {}
// PROJECTION-ERROR-NEXT: "final_outputs": []
// PROJECTION-ERROR-NOT: "dataflow.load":
// PROJECTION-ERROR-NOT: "dataflow.store":
// PROJECTION-ERROR: "status": "unsupported"

// The same element under an explicit done/ctrl chain is ordered, so both
// accesses fire and the load observes the stored value.
// CHAIN: "final_outputs": [
// CHAIN-NEXT: "none",
// CHAIN-NEXT: "i32:42"
// CHAIN: "dataflow.load": 1
// CHAIN: "dataflow.store": 1
// CHAIN: "status": "pass"

// A load result used as store data is an ordinary SSA dependency, not
// canonical memory order. With both controls still rooted at the same start
// token, the overlapping store remains unordered even though it cannot become
// ready until the load publishes data. Only load done feeding store ctrl would
// establish the required order.
// DATA-DEPENDENCY: "final_memory_roots": {}
// DATA-DEPENDENCY-NEXT: "final_memory_state": {}
// DATA-DEPENDENCY-NEXT: "final_outputs": []
// DATA-DEPENDENCY: "dataflow.load": 1
// DATA-DEPENDENCY-NOT: "dataflow.store":
// DATA-DEPENDENCY: "status": "unsupported"

// An all-zero masked store performs no memory action. Its done token may carry
// only its ctrl frontier, so using load data as its inactive payload cannot
// launder the load effect into a later store's ctrl. The later overlapping
// store remains unordered from the load.
// EMPTY-STORE: "final_memory_roots": {}
// EMPTY-STORE-NEXT: "final_memory_state": {}
// EMPTY-STORE-NEXT: "final_outputs": []
// EMPTY-STORE: "dataflow.load": 1
// EMPTY-STORE: "dataflow.store": 1
// EMPTY-STORE: "status": "unsupported"

// Two unordered plain loads of one element do not conflict.
// RR: "final_outputs": [
// RR-NEXT: "none",
// RR-NEXT: "i32:15",
// RR-NEXT: "i32:15"
// RR: "dataflow.load": 2
// RR: "status": "pass"

// Two unordered plain stores to disjoint elements do not conflict and both
// commit.
// DISJOINT: "arg0": [
// DISJOINT-NEXT: "i32:100",
// DISJOINT-NEXT: "i32:11",
// DISJOINT-NEXT: "i32:12",
// DISJOINT-NEXT: "i32:13",
// DISJOINT-NEXT: "i32:14",
// DISJOINT-NEXT: "i32:15",
// DISJOINT-NEXT: "i32:16",
// DISJOINT-NEXT: "i32:200"
// DISJOINT: "dataflow.store": 2
// DISJOINT: "status": "pass"

// Lane 0 of the masked scatter addresses element 2 but is inactive, so it
// derives no access and cannot conflict with the unordered load of that
// element. Only the active lane 1 commits, to element 5. The inactive lane
// carries a different value, so an access derived from it would be visible in
// both the load result and element 2.
// MASKED: "arg3": [
// MASKED-NEXT: "i32:10",
// MASKED-NEXT: "i32:11",
// MASKED-NEXT: "i32:12",
// MASKED-NEXT: "i32:13",
// MASKED-NEXT: "i32:14",
// MASKED-NEXT: "i32:99",
// MASKED-NEXT: "i32:16",
// MASKED-NEXT: "i32:17"
// MASKED: "final_outputs": [
// MASKED-NEXT: "none",
// MASKED-NEXT: "i32:12"
// MASKED: "dataflow.load": 1
// MASKED: "dataflow.store": 1
// MASKED: "status": "pass"

// A group actor assembles its vector across several firings after a load orders
// the stream activation. The completed group must preserve that explicit
// frontier through both stateful actors. The vector store then writes the two
// induction values into elements 2 and 3.
// GROUP: "arg0": [
// GROUP-NEXT: "i8:10",
// GROUP-NEXT: "i8:11",
// GROUP-NEXT: "i8:0",
// GROUP-NEXT: "i8:1",
// GROUP-NEXT: "i8:14",
// GROUP-NEXT: "i8:15",
// GROUP-NEXT: "i8:16",
// GROUP-NEXT: "i8:17"
// GROUP: "dataflow.load": 1
// GROUP: "dataflow.store": 1
// GROUP: "status": "pass"

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

  dataflow.graph private @unordered_ww_overlap_reversed(
      %start: none, %packed: i64, %mem: memref<8xi32>) -> ()
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data = dataflow.unpack %packed : i64 -> vector<2xi32>
    %base = dataflow.constant %start {const_value = 1 : index} : index
    %cell = dataflow.constant %start {const_value = 2 : index} : index
    %value = dataflow.constant %start {const_value = 200 : i32} : i32
    %scalar_done = dataflow.store %mem[%cell] %value %start : memref<8xi32>
    %vector_done = dataflow.store %mem[%base] %data %start
        : memref<8xi32>, vector<2xi32>
    dataflow.graph.return values() streams() memories()
        complete(%vector_done, %scalar_done : none, none)
  }

  dataflow.graph private @uninitialized_unordered_rw_precedence(
      %start: none) -> (memref<1xi32>)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 1>} {
    %slot = memref.alloc() : memref<1xi32>
    %cell = dataflow.constant %start {const_value = 0 : index} : index
    %value = dataflow.constant %start {const_value = 42 : i32} : i32
    %store_done = dataflow.store %slot[%cell] %value %start : memref<1xi32>
    %loaded, %load_done = dataflow.load %slot[%cell] %start : memref<1xi32>
    dataflow.graph.return values() streams()
        memories(%slot : memref<1xi32>)
        complete(%store_done, %load_done : none, none)
  }

  dataflow.graph private @conflict_precedes_projection_error(
      %start: none, %mem: memref<1xi32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %bad = dataflow.constant %start {const_value = 7 : index} : index
    %cell = dataflow.constant %start {const_value = 0 : index} : index
    %first = dataflow.constant %start {const_value = 41 : i32} : i32
    %second = dataflow.constant %start {const_value = 42 : i32} : i32
    %unused, %bad_done = dataflow.load %mem[%bad] %start : memref<1xi32>
    %first_done = dataflow.store %mem[%cell] %first %start : memref<1xi32>
    %second_done = dataflow.store %mem[%cell] %second %start : memref<1xi32>
    dataflow.graph.return values() streams() memories()
        complete(%bad_done, %first_done, %second_done : none, none, none)
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

  dataflow.graph private @data_dependency_does_not_order_memory(
      %start: none, %mem: memref<1xi32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %cell = dataflow.constant %start {const_value = 0 : index} : index
    %loaded, %load_done = dataflow.load %mem[%cell] %start : memref<1xi32>
    %store_done = dataflow.store %mem[%cell] %loaded %start : memref<1xi32>
    dataflow.graph.return values() streams() memories()
        complete(%load_done, %store_done : none, none)
  }

  dataflow.graph private @empty_store_does_not_launder_order(
      %start: none, %mem: memref<1xi32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %cell = dataflow.constant %start {const_value = 0 : index} : index
    %zero = dataflow.constant %start {const_value = 0 : i1} : i1
    %later = dataflow.constant %start {const_value = 99 : i32} : i32
    %loaded, %load_done = dataflow.load %mem[%cell] %start : memref<1xi32>
    %inactive_data = dataflow.unpack %loaded : i32 -> vector<1xi32>
    %mask = dataflow.unpack %zero : i1 -> vector<1xi1>
    %empty_done = dataflow.store %mem[%cell] %inactive_data %start mask %mask
        : memref<1xi32>, vector<1xi32>
    %later_done = dataflow.store %mem[%cell] %later %empty_done
        : memref<1xi32>
    dataflow.graph.return values() streams() memories()
        complete(%load_done, %later_done : none, none)
  }

  dataflow.graph private @unordered_rr_overlap(
      %start: none, %mem: memref<8xi32>) -> (i32, i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %cell = dataflow.constant %start {const_value = 5 : index} : index
    %first, %first_done = dataflow.load %mem[%cell] %start : memref<8xi32>
    %second, %second_done = dataflow.load %mem[%cell] %start : memref<8xi32>
    dataflow.graph.return values(%first, %second : i32, i32) streams()
        memories() complete(%first_done, %second_done : none, none)
  }

  dataflow.graph private @unordered_ww_disjoint(
      %start: none, %mem: memref<8xi32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %lo = dataflow.constant %start {const_value = 0 : index} : index
    %hi = dataflow.constant %start {const_value = 7 : index} : index
    %lo_value = dataflow.constant %start {const_value = 100 : i32} : i32
    %hi_value = dataflow.constant %start {const_value = 200 : i32} : i32
    %lo_done = dataflow.store %mem[%lo] %lo_value %start : memref<8xi32>
    %hi_done = dataflow.store %mem[%hi] %hi_value %start : memref<8xi32>
    dataflow.graph.return values() streams() memories()
        complete(%lo_done, %hi_done : none, none)
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

  dataflow.graph private @parallelize_group_inherits_lane_order(
      %start: none, %mem: memref<8xi8>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %cell = dataflow.constant %start {const_value = 2 : index} : index
    %loaded, %load_done = dataflow.load %mem[%cell] %start : memref<8xi8>
    %lb = dataflow.constant %start {const_value = 0 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %two = dataflow.constant %start {const_value = 2 : i8} : i8
    %bound:2 = dataflow.sync %load_done, %two : (none, i8) -> (none, i8)
    %item, %scalar_phase = dataflow.stream %lb, %bound#1, %one
        step add while ult : i8
    %vector, %mask, %group_phase = dataflow.parallelize %item, %scalar_phase
        : (i8, i1) -> (vector<2xi8>, vector<2xi1>, i1)
    %group_cell = dataflow.invariant %group_phase, %cell : index
    %group_units = dataflow.invariant %group_phase, %start : none
    %group_events:2 = dataflow.demux %group_phase, %group_units
        : (i1, none) -> (none, none)
    %store_done = dataflow.store %mem[%group_cell] %vector %group_events#1
        mask %mask : memref<8xi8>, vector<2xi8>
    %memory_frontier = dataflow.carry %group_phase, %start, %store_done : none
    %memory_lane:2 = dataflow.demux %group_phase, %memory_frontier
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%memory_lane#0 : none)
  }

  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
  } {
    dataflow.graph private @unordered_rw_overlap(
        %start: none, %addresses: vector<2xindex>, %mem: memref<8xi32>) -> ()
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 0, 0, 0>} {
      %cell = dataflow.constant %start {const_value = 3 : index} : index
      %value = dataflow.constant %start {const_value = 77 : i32} : i32
      %store_done = dataflow.store %mem[%cell] %value %start : memref<8xi32>
      %data, %load_done = dataflow.load %mem[%addresses] %start
          : memref<8xi32>, vector<2xindex>, vector<2xi32>
      dataflow.graph.return values() streams() memories()
          complete(%store_done, %load_done : none, none)
    }

    dataflow.graph private @masked_inactive_lane_no_access(
        %start: none, %packed: i64, %addresses: vector<2xindex>,
        %packed_mask: i2, %mem: memref<8xi32>) -> (i32)
        attributes {input_segments = array<i32: 3, 0, 1>,
                    result_segments = array<i32: 1, 0, 0>} {
      %data = dataflow.unpack %packed : i64 -> vector<2xi32>
      %mask = dataflow.unpack %packed_mask : i2 -> vector<2xi1>
      %cell = dataflow.constant %start {const_value = 2 : index} : index
      %store_done = dataflow.store %mem[%addresses] %data %start mask %mask
          : memref<8xi32>, vector<2xindex>, vector<2xi32>
      %loaded, %load_done = dataflow.load %mem[%cell] %start : memref<8xi32>
      dataflow.graph.return values(%loaded : i32) streams() memories()
          complete(%store_done, %load_done : none, none)
    }
  }
}
