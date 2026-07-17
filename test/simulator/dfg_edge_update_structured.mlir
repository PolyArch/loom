// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph edge_update_structured --arg 0=2 --arg 1=4 --arg 2=100 --arg 3=8 --arg 4=16 --memref 5=0,2,4,7,10,12,14,15,16 --memref 6=1,2,0,3,0,4,5,1,2,6,3,7,4,6,7,5 --memref 7=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --memref 8=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --output %t.update.json
// RUN: FileCheck %s --check-prefix=UPDATE < %t.update.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph edge_update_find_slot --arg 0=4 --arg 1=7 --arg 2=4 --memref 3=1,2,0,3,0,4,5,1,2,6,3,7,4,6,7,5 --output %t.find.json
// RUN: FileCheck %s --check-prefix=FIND < %t.find.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph effect_switch_store --arg 0=2 --arg 1=5 --arg 2=100 --memref 3=0,0,0,0,0,0,0,0 --output %t.switch-store.json
// RUN: FileCheck %s --check-prefix=SWITCH-STORE < %t.switch-store.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph nested_if_switch_store --arg 0=2 --arg 1=5 --arg 2=100 --arg 3=true --memref 4=0,0,0,0,0,0,0,0 --output %t.nested-store.json
// RUN: FileCheck %s --check-prefix=NESTED-STORE < %t.nested-store.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph while_driven_switch_store --arg 0=4 --arg 1=7 --arg 2=4 --arg 3=100 --arg 4=true --memref 5=1,2,0,3,0,4,5,1,2,6,3,7,4,6,7,5 --memref 6=0,0,0,0,0,0,0,0 --output %t.while-store.json
// RUN: FileCheck %s --check-prefix=WHILE-STORE < %t.while-store.json

// UPDATE: "final_memory_state": {
// UPDATE: "arg8": [
// UPDATE-NEXT: "i32:1",
// UPDATE-NEXT: "i32:2",
// UPDATE-NEXT: "i32:3",
// UPDATE-NEXT: "i32:4",
// UPDATE-NEXT: "i32:5",
// UPDATE-NEXT: "i32:100",
// UPDATE-NEXT: "i32:7",
// UPDATE: "graph": "edge_update_structured"
// UPDATE: "operation_fire_counts": {
// UPDATE-DAG: "dataflow.stream": 17
// UPDATE-DAG: "dataflow.demux": 95
// UPDATE-DAG: "dataflow.mux": 17
// UPDATE-DAG: "dataflow.load": 20
// UPDATE-DAG: "dataflow.store": 17
// UPDATE: "status": "pass"
// UPDATE: "workload": "edge_update_structured"

// FIND: "final_outputs": [
// FIND-NEXT: "none",
// FIND-NEXT: "i64:5",
// FIND-NEXT: "i32:2"
// FIND: "status": "pass"

// SWITCH-STORE: "final_memory_state": {
// SWITCH-STORE: "arg3": [
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:100",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE: "arith.cmpi": 1
// SWITCH-STORE: "dataflow.demux": 5
// SWITCH-STORE: "dataflow.mux": 2
// SWITCH-STORE: "dataflow.store": 1
// SWITCH-STORE: "status": "pass"

// NESTED-STORE: "final_memory_state": {
// NESTED-STORE: "arg4": [
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:100",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE: "arith.cmpi": 1
// NESTED-STORE: "dataflow.demux": 12
// NESTED-STORE: "dataflow.mux": 5
// NESTED-STORE: "dataflow.store": 1
// NESTED-STORE: "status": "pass"

// WHILE-STORE: "final_memory_state": {
// WHILE-STORE: "arg6": [
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:100",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-DAG: "arith.cmpi": 7
// WHILE-STORE-DAG: "dataflow.demux": 28
// WHILE-STORE-DAG: "dataflow.mux": 11
// WHILE-STORE-DAG: "dataflow.load": 2
// WHILE-STORE-DAG: "dataflow.store": 1
// WHILE-STORE: "status": "pass"

module {
  dataflow.graph.func private @while_driven_switch_store(
      %ctrl: none, %row_begin: i32, %row_end: i32, %dst: i32, %value: i32,
      %enabled: i1, %cols: !llvm.ptr, %output: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 5, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %two = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %zero_0 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    scf.if %enabled {
      %found:3 = scf.while (%cursor = %row_begin) : (i32) -> (i32, i64, i32) {
        %cursor_i64 = llvm.zext %cursor : i32 to i64
        %cols_view = builtin.unrealized_conversion_cast %cols
            : !llvm.ptr to memref<?xi32>
        %cursor_index = arith.index_cast %cursor : i32 to index
        %col, %col_done = dataflow.load %cols_view[%cursor_index] %ctrl
            : memref<?xi32>
        %matches = arith.cmpi eq, %col, %dst : i32
        %next = arith.addi %cursor, %one : i32
        %at_end = arith.cmpi eq, %next, %row_end : i32
        %at_end_i32 = arith.extui %at_end : i1 to i32
        %before_end = arith.cmpi ne, %next, %row_end : i32
        %before_end_i32 = arith.extui %before_end : i1 to i32
        %next_cursor = dataflow.mux %matches, %next, %zero
            : (i1, i32, i32) -> i32
        %done_code = dataflow.mux %matches, %at_end_i32, %two
            : (i1, i32, i32) -> i32
        %continue_code = dataflow.mux %matches, %before_end_i32, %zero_0
            : (i1, i32, i32) -> i32
        %keep_running = arith.trunci %continue_code : i32 to i1
        scf.condition(%keep_running) %next_cursor, %cursor_i64, %done_code
            : i32, i64, i32
      } do {
      ^bb0(%cursor: i32, %write_index: i64, %status: i32):
        scf.yield %cursor : i32
      }
      %skip_store = arith.cmpi eq, %found#2, %one : i32
      scf.if %skip_store {
      } else {
        %output_view = builtin.unrealized_conversion_cast %output
            : !llvm.ptr to memref<?xi32>
        %slot = arith.index_cast %found#1 : i64 to index
        %store_done = dataflow.store %output_view[%slot] %value %ctrl
            : memref<?xi32>
      }
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @nested_if_switch_store(
      %ctrl: none, %selector: index, %slot: index, %value: i32, %enabled: i1,
      %output: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %one = dataflow.constant %ctrl {const_value = 1 : index} : index
    scf.if %enabled {
      %skip_store = arith.cmpi eq, %selector, %one : index
      scf.if %skip_store {
      } else {
        %output_view = builtin.unrealized_conversion_cast %output
            : !llvm.ptr to memref<?xi32>
        %store_done = dataflow.store %output_view[%slot] %value %ctrl
            : memref<?xi32>
      }
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @effect_switch_store(
      %ctrl: none, %selector: index, %slot: index, %value: i32,
      %output: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %one = dataflow.constant %ctrl {const_value = 1 : index} : index
    %skip_store = arith.cmpi eq, %selector, %one : index
    scf.if %skip_store {
    } else {
      %output_view = builtin.unrealized_conversion_cast %output
          : !llvm.ptr to memref<?xi32>
      %store_done = dataflow.store %output_view[%slot] %value %ctrl
          : memref<?xi32>
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @edge_update_find_slot(
      %ctrl: none, %row_begin: i32, %row_end: i32, %dst: i32,
      %cols: !llvm.ptr) -> (none, i64, i32)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %two = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %zero_0 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %found:3 = scf.while (%cursor = %row_begin) : (i32) -> (i32, i64, i32) {
      %cursor_i64 = llvm.zext %cursor : i32 to i64
      %cols_view = builtin.unrealized_conversion_cast %cols
          : !llvm.ptr to memref<?xi32>
      %cursor_index = arith.index_cast %cursor : i32 to index
      %col, %col_done = dataflow.load %cols_view[%cursor_index] %ctrl
          : memref<?xi32>
      %matches = arith.cmpi eq, %col, %dst : i32
      %next = arith.addi %cursor, %one : i32
      %at_end = arith.cmpi eq, %next, %row_end : i32
      %at_end_i32 = arith.extui %at_end : i1 to i32
      %before_end = arith.cmpi ne, %next, %row_end : i32
      %before_end_i32 = arith.extui %before_end : i1 to i32
      %next_cursor = dataflow.mux %matches, %next, %zero
          : (i1, i32, i32) -> i32
      %done_code = dataflow.mux %matches, %at_end_i32, %two
          : (i1, i32, i32) -> i32
      %continue_code = dataflow.mux %matches, %before_end_i32, %zero_0
          : (i1, i32, i32) -> i32
      %keep_running = arith.trunci %continue_code : i32 to i1
      scf.condition(%keep_running) %next_cursor, %cursor_i64, %done_code
          : i32, i64, i32
    } do {
    ^bb0(%cursor: i32, %write_index: i64, %status: i32):
      scf.yield %cursor : i32
    }
    dataflow.graph.return %ctrl, %found#1, %found#2 : none, i64, i32
  }

  dataflow.graph.func private @edge_update_structured(
      %ctrl: none, %src: i32, %dst: i32, %new_weight: i32, %nodes: i32,
      %edges: i32, %row_ptr: !llvm.ptr, %cols: !llvm.ptr, %input: !llvm.ptr,
      %output: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 5, 0, 4>,
                  result_segments = array<i32: 0, 0, 0>} {
    %two = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %zero_0 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %zero_index = dataflow.constant %ctrl {const_value = 0 : index} : index
    %step_index = dataflow.constant %ctrl {const_value = 1 : index} : index
    %no_edges = arith.cmpi eq, %edges, %zero_0 : i32
    scf.if %no_edges {
    } else {
      %extent = arith.index_cast %edges : i32 to index
      scf.for %i = %zero_index to %extent step %step_index {
        %input_view = builtin.unrealized_conversion_cast %input
            : !llvm.ptr to memref<?xi32>
        %value, %load_done = dataflow.load %input_view[%i] %ctrl
            : memref<?xi32>
        %output_view = builtin.unrealized_conversion_cast %output
            : !llvm.ptr to memref<?xi32>
        %store_done = dataflow.store %output_view[%i] %value %ctrl
            : memref<?xi32>
      }
    }
    %valid_src = arith.cmpi ult, %src, %nodes : i32
    scf.if %valid_src {
      %row_view = builtin.unrealized_conversion_cast %row_ptr
          : !llvm.ptr to memref<?xi32>
      %src_index = arith.index_cast %src : i32 to index
      %row_begin, %row_begin_done = dataflow.load %row_view[%src_index] %ctrl
          : memref<?xi32>
      %row_view_0 = builtin.unrealized_conversion_cast %row_ptr
          : !llvm.ptr to memref<?xi32>
      %src_index_0 = arith.index_cast %src : i32 to index
      %one_index = arith.index_cast %one : i32 to index
      %next_src_index = arith.addi %src_index_0, %one_index : index
      %row_end, %row_end_done = dataflow.load %row_view_0[%next_src_index] %ctrl
          : memref<?xi32>
      %has_edges = arith.cmpi ult, %row_begin, %row_end : i32
      scf.if %has_edges {
        %found:3 = scf.while (%cursor = %row_begin) : (i32) -> (i32, i64, i32) {
          %cursor_i64 = llvm.zext %cursor : i32 to i64
          %cols_view = builtin.unrealized_conversion_cast %cols
              : !llvm.ptr to memref<?xi32>
          %cursor_index = arith.index_cast %cursor : i32 to index
          %col, %col_done = dataflow.load %cols_view[%cursor_index] %ctrl
              : memref<?xi32>
          %matches = arith.cmpi eq, %col, %dst : i32
          %next = arith.addi %cursor, %one : i32
          %at_end = arith.cmpi eq, %next, %row_end : i32
          %at_end_i32 = arith.extui %at_end : i1 to i32
          %before_end = arith.cmpi ne, %next, %row_end : i32
          %before_end_i32 = arith.extui %before_end : i1 to i32
          %next_cursor = dataflow.mux %matches, %next, %zero
              : (i1, i32, i32) -> i32
          %done_code = dataflow.mux %matches, %at_end_i32, %two
              : (i1, i32, i32) -> i32
          %continue_code = dataflow.mux %matches, %before_end_i32, %zero_0
              : (i1, i32, i32) -> i32
          %keep_running = arith.trunci %continue_code : i32 to i1
          scf.condition(%keep_running) %next_cursor, %cursor_i64, %done_code
              : i32, i64, i32
        } do {
        ^bb0(%cursor: i32, %write_index: i64, %status: i32):
          scf.yield %cursor : i32
        }
        %skip_store = arith.cmpi eq, %found#2, %one : i32
        scf.if %skip_store {
        } else {
          %output_view = builtin.unrealized_conversion_cast %output
              : !llvm.ptr to memref<?xi32>
          %slot = arith.index_cast %found#1 : i64 to index
          %store_done = dataflow.store %output_view[%slot] %new_weight %ctrl
              : memref<?xi32>
        }
      }
    }
    dataflow.graph.return %ctrl : none
  }
}
