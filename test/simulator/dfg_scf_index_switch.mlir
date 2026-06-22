// RUN: loom-dfg-sim %s --graph index_switch_scalar --arg 0=none --arg 1=2 --output %t.scalar.json
// RUN: FileCheck %s --check-prefix=SCALAR < %t.scalar.json
// RUN: loom-dfg-sim %s --graph structured_for_index_switch --arg 0=none --arg 1=0 --arg 2=3 --arg 3=1 --arg 4=0 --output %t.for.json
// RUN: FileCheck %s --check-prefix=FOR < %t.for.json
// RUN: loom-dfg-sim %s --graph structured_for_merge_switch --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=3 --arg 5=true --arg 6=1 --arg 7=0 --memref 8=1,4,9 --memref 9=2,3,10,14 --memref 10=0,0,0,0,0 --output %t.merge.json
// RUN: FileCheck %s --check-prefix=MERGE < %t.merge.json

// SCALAR-DAG: "workload": "index_switch_scalar"
// SCALAR-DAG: "graph": "index_switch_scalar"
// SCALAR-DAG: "status": "pass"
// SCALAR-DAG: "scf.index_switch": 1
// SCALAR-DAG: "final_outputs": [
// SCALAR-DAG: "none"
// SCALAR-DAG: "i32:20"

// FOR-DAG: "workload": "structured_for_index_switch"
// FOR-DAG: "graph": "structured_for_index_switch"
// FOR-DAG: "status": "pass"
// FOR-DAG: "dynamic_work_items": 3
// FOR-DAG: "scf.index_switch": 3
// FOR-DAG: "final_outputs": [
// FOR-DAG: "none"
// FOR-DAG: "i32:4"

// MERGE-DAG: "workload": "structured_for_merge_switch"
// MERGE-DAG: "graph": "structured_for_merge_switch"
// MERGE-DAG: "status": "pass"
// MERGE-DAG: "dynamic_work_items": 4
// MERGE-DAG: "scf.if": 8
// MERGE-DAG: "scf.index_switch": 8
// MERGE-DAG: "dataflow.load": 12
// MERGE-DAG: "dataflow.store": 4
// MERGE-DAG: "final_outputs": [
// MERGE-DAG: "none"
// MERGE-DAG: "i32:2"
// MERGE-DAG: "i32:2"
// MERGE-DAG: "arg10": [
// MERGE-DAG: "f32:1"
// MERGE-DAG: "f32:2"
// MERGE-DAG: "f32:3"
// MERGE-DAG: "f32:4"

module {
  dataflow.graph.func private @index_switch_scalar(%ctrl: none, %selector: index)
      -> (none, i32) {
    %value = scf.index_switch %selector -> i32
    case 1 {
      %ten = arith.constant 10 : i32
      scf.yield %ten : i32
    }
    case 2 {
      %twenty = arith.constant 20 : i32
      scf.yield %twenty : i32
    }
    default {
      %fallback = arith.constant 30 : i32
      scf.yield %fallback : i32
    }
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @structured_for_index_switch(
      %ctrl: none, %lb: index, %ub: index, %step: index, %init: i32)
      -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init)
        -> (i32) {
      %addend = scf.index_switch %i -> i32
      case 1 {
        %two = arith.constant 2 : i32
        scf.yield %two : i32
      }
      default {
        %one = arith.constant 1 : i32
        scf.yield %one : i32
      }
      %next = arith.addi %acc, %addend : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_for_merge_switch(
      %ctrl: none, %lb: i64, %ub: i64, %step: i64, %limit: i32, %invert: i1,
      %one: i32, %zero: i32, %lhs_ptr: !llvm.ptr, %rhs_ptr: !llvm.ptr,
      %dst_ptr: !llvm.ptr) -> (none, i32, i32) {
    %cursors:2 = scf.for %i = %lb to %ub step %step
        iter_args(%lhs_i = %zero, %rhs_i = %zero) -> (i32, i32) : i64 {
      %rhs_done = arith.cmpi ugt, %rhs_i, %limit : i32
      %choice:3 = scf.if %rhs_done -> (i32, i32, i32) {
        scf.yield %zero, %zero, %zero : i32, i32, i32
      } else {
        %has_lhs = arith.cmpi ult, %lhs_i, %limit : i32
        %selector_i32 = scf.if %has_lhs -> (i32) {
          %lhs_mem = builtin.unrealized_conversion_cast %lhs_ptr
              : !llvm.ptr to memref<?xf32>
          %lhs_slot = arith.index_cast %lhs_i : i32 to index
          %lhs_value, %lhs_done = dataflow.load %lhs_mem[%lhs_slot] %ctrl
              : memref<?xf32>
          %rhs_mem = builtin.unrealized_conversion_cast %rhs_ptr
              : !llvm.ptr to memref<?xf32>
          %rhs_slot = arith.index_cast %rhs_i : i32 to index
          %rhs_value, %rhs_done_0 = dataflow.load %rhs_mem[%rhs_slot] %ctrl
              : memref<?xf32>
          %lhs_greater = arith.cmpf ugt, %lhs_value, %rhs_value : f32
          %take_rhs = arith.xori %lhs_greater, %invert : i1
          %selector = arith.extui %take_rhs : i1 to i32
          scf.yield %selector : i32
        } else {
          scf.yield %zero : i32
        }
        %selector_index = arith.index_castui %selector_i32 : i32 to index
        %rhs_choice:3 = scf.index_switch %selector_index -> i32, i32, i32
        case 0 {
          %rhs_mem = builtin.unrealized_conversion_cast %rhs_ptr
              : !llvm.ptr to memref<?xf32>
          %rhs_slot = arith.index_cast %rhs_i : i32 to index
          %rhs_value, %rhs_done_1 = dataflow.load %rhs_mem[%rhs_slot] %ctrl
              : memref<?xf32>
          %dst_mem = builtin.unrealized_conversion_cast %dst_ptr
              : !llvm.ptr to memref<?xf32>
          %dst_slot = arith.index_cast %i : i64 to index
          %stored = dataflow.store %dst_mem[%dst_slot] %rhs_value %ctrl
              : memref<?xf32>
          %next_rhs = arith.addi %rhs_i, %one : i32
          scf.yield %next_rhs, %lhs_i, %one : i32, i32, i32
        }
        default {
          scf.yield %zero, %zero, %zero : i32, i32, i32
        }
        scf.yield %rhs_choice#0, %rhs_choice#1, %rhs_choice#2
            : i32, i32, i32
      }
      %outer_selector = arith.index_castui %choice#2 : i32 to index
      %next:2 = scf.index_switch %outer_selector -> i32, i32
      case 0 {
        %lhs_mem = builtin.unrealized_conversion_cast %lhs_ptr
            : !llvm.ptr to memref<?xf32>
        %lhs_slot = arith.index_cast %lhs_i : i32 to index
        %lhs_value, %lhs_done_1 = dataflow.load %lhs_mem[%lhs_slot] %ctrl
            : memref<?xf32>
        %dst_mem = builtin.unrealized_conversion_cast %dst_ptr
            : !llvm.ptr to memref<?xf32>
        %dst_slot = arith.index_cast %i : i64 to index
        %stored = dataflow.store %dst_mem[%dst_slot] %lhs_value %ctrl
            : memref<?xf32>
        %next_lhs = arith.addi %lhs_i, %one : i32
        scf.yield %rhs_i, %next_lhs : i32, i32
      }
      default {
        scf.yield %choice#0, %choice#1 : i32, i32
      }
      scf.yield %next#1, %next#0 : i32, i32
    }
    dataflow.graph.return %ctrl, %cursors#0, %cursors#1 : none, i32, i32
  }
}
