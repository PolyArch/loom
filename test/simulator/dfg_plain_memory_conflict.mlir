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
}
