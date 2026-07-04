// RUN: loom-dfg-sim %s --graph edge_update_structured --arg 0=none --memref 1=0,2,4,7,10,12,14,15,16 --memref 2=1,2,0,3,0,4,5,1,2,6,3,7,4,6,7,5 --memref 3=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --memref 4=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --arg 5=2 --arg 6=4 --arg 7=100 --arg 8=8 --arg 9=16 --output %t.update.json
// RUN: FileCheck %s --check-prefix=UPDATE < %t.update.json
// RUN: loom-dfg-sim %s --graph edge_update_find_slot --arg 0=none --memref 1=1,2,0,3,0,4,5,1,2,6,3,7,4,6,7,5 --arg 2=4 --arg 3=7 --arg 4=4 --output %t.find.json
// RUN: FileCheck %s --check-prefix=FIND < %t.find.json
// RUN: loom-dfg-sim %s --graph effect_switch_store --arg 0=none --memref 1=0,0,0,0,0,0,0,0 --arg 2=2 --arg 3=5 --arg 4=100 --output %t.switch-store.json
// RUN: FileCheck %s --check-prefix=SWITCH-STORE < %t.switch-store.json
// RUN: loom-dfg-sim %s --graph nested_if_switch_store --arg 0=none --memref 1=0,0,0,0,0,0,0,0 --arg 2=2 --arg 3=5 --arg 4=100 --arg 5=true --output %t.nested-store.json
// RUN: FileCheck %s --check-prefix=NESTED-STORE < %t.nested-store.json
// RUN: loom-dfg-sim %s --graph while_driven_switch_store --arg 0=none --memref 1=1,2,0,3,0,4,5,1,2,6,3,7,4,6,7,5 --memref 2=0,0,0,0,0,0,0,0 --arg 3=4 --arg 4=7 --arg 5=4 --arg 6=100 --arg 7=true --output %t.while-store.json
// RUN: FileCheck %s --check-prefix=WHILE-STORE < %t.while-store.json

// UPDATE: "final_memory_state": {
// UPDATE: "arg4": [
// UPDATE-NEXT: "i32:1",
// UPDATE-NEXT: "i32:2",
// UPDATE-NEXT: "i32:3",
// UPDATE-NEXT: "i32:4",
// UPDATE-NEXT: "i32:5",
// UPDATE-NEXT: "i32:100",
// UPDATE-NEXT: "i32:7",
// UPDATE: "graph": "edge_update_structured"
// UPDATE: "operation_fire_counts": {
// UPDATE-DAG: "scf.if": 3
// UPDATE-DAG: "scf.index_switch": 1
// UPDATE-DAG: "dataflow.store": 17
// UPDATE: "status": "pass"
// UPDATE: "workload": "edge_update_structured"

// FIND: "final_outputs": [
// FIND-NEXT: "none",
// FIND-NEXT: "i64:5",
// FIND-NEXT: "i32:2"
// FIND: "status": "pass"

// SWITCH-STORE: "final_memory_state": {
// SWITCH-STORE: "arg1": [
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE-NEXT: "i32:100",
// SWITCH-STORE-NEXT: "i32:0",
// SWITCH-STORE: "scf.index_switch": 1
// SWITCH-STORE: "status": "pass"

// NESTED-STORE: "final_memory_state": {
// NESTED-STORE: "arg1": [
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE-NEXT: "i32:100",
// NESTED-STORE-NEXT: "i32:0",
// NESTED-STORE: "scf.if": 1
// NESTED-STORE: "scf.index_switch": 1
// NESTED-STORE: "status": "pass"

// WHILE-STORE: "final_memory_state": {
// WHILE-STORE: "arg2": [
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE-NEXT: "i32:100",
// WHILE-STORE-NEXT: "i32:0",
// WHILE-STORE: "scf.if": 1
// WHILE-STORE: "scf.index_switch": 1
// WHILE-STORE: "status": "pass"

module {
  dataflow.graph.func private @while_driven_switch_store(
      %ctrl: none, %cols: !llvm.ptr, %output: !llvm.ptr, %row_begin: i32,
      %row_end: i32, %dst: i32, %value: i32, %enabled: i1) -> none {
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
      %status_index = arith.index_castui %found#2 : i32 to index
      scf.index_switch %status_index
      case 1 {
        scf.yield
      }
      default {
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
      %ctrl: none, %output: !llvm.ptr, %selector: index, %slot: index,
      %value: i32, %enabled: i1) -> none {
    scf.if %enabled {
      scf.index_switch %selector
      case 1 {
        scf.yield
      }
      default {
        %output_view = builtin.unrealized_conversion_cast %output
            : !llvm.ptr to memref<?xi32>
        %store_done = dataflow.store %output_view[%slot] %value %ctrl
            : memref<?xi32>
      }
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @effect_switch_store(
      %ctrl: none, %output: !llvm.ptr, %selector: index, %slot: index,
      %value: i32) -> none {
    scf.index_switch %selector
    case 1 {
      scf.yield
    }
    default {
      %output_view = builtin.unrealized_conversion_cast %output
          : !llvm.ptr to memref<?xi32>
      %store_done = dataflow.store %output_view[%slot] %value %ctrl
          : memref<?xi32>
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @edge_update_find_slot(
      %ctrl: none, %cols: !llvm.ptr, %row_begin: i32, %row_end: i32,
      %dst: i32) -> (none, i64, i32) {
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
      %ctrl: none, %row_ptr: !llvm.ptr, %cols: !llvm.ptr, %input: !llvm.ptr,
      %output: !llvm.ptr, %src: i32, %dst: i32, %new_weight: i32,
      %nodes: i32, %edges: i32) -> none {
    %two = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %zero_0 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %no_edges = arith.cmpi eq, %edges, %zero_0 : i32
    scf.if %no_edges {
    } else {
      %extent = arith.index_cast %edges : i32 to index
      scf.forall (%i) in (%extent) {
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
        %status_index = arith.index_castui %found#2 : i32 to index
        scf.index_switch %status_index
        case 1 {
          scf.yield
        }
        default {
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
