// RUN: loom-dfg-sim %s --graph compare_select --arg 0=none --output %t.compare.json
// RUN: FileCheck %s --check-prefix=COMPARE < %t.compare.json
// RUN: loom-dfg-sim %s --graph integer_mix --arg 0=none --output %t.integer.json
// RUN: FileCheck %s --check-prefix=INTEGER < %t.integer.json
// RUN: loom-dfg-sim %s --graph byte_swap --arg 0=none --output %t.bswap.json
// RUN: FileCheck %s --check-prefix=BSWAP < %t.bswap.json
// RUN: loom-dfg-sim %s --graph zext_bits --arg 0=none --output %t.zext.json
// RUN: FileCheck %s --check-prefix=ZEXT < %t.zext.json

// COMPARE-DAG: "workload": "compare_select"
// COMPARE-DAG: "graph": "compare_select"
// COMPARE-DAG: "status": "pass"
// COMPARE-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// COMPARE-DAG: "optimistic_cycles": 5
// COMPARE-DAG: "event_count": 4
// COMPARE-DAG: "f32:3"

// INTEGER-DAG: "workload": "integer_mix"
// INTEGER-DAG: "graph": "integer_mix"
// INTEGER-DAG: "status": "pass"
// INTEGER-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// INTEGER-DAG: "optimistic_cycles": 21
// INTEGER-DAG: "event_count": 14
// INTEGER-DAG: "i32:3"

// BSWAP-DAG: "workload": "byte_swap"
// BSWAP-DAG: "graph": "byte_swap"
// BSWAP-DAG: "status": "pass"
// BSWAP-DAG: "optimistic_cycles": 2
// BSWAP-DAG: "event_count": 2
// BSWAP-DAG: "i32:2018915346"

// ZEXT-DAG: "workload": "zext_bits"
// ZEXT-DAG: "graph": "zext_bits"
// ZEXT-DAG: "status": "pass"
// ZEXT-DAG: "optimistic_cycles": 2
// ZEXT-DAG: "i64:4294967295"

module {
  dataflow.graph.func private @compare_select(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 9.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %pred = arith.cmpf ugt, %lhs, %rhs : f32
    %selected = llvm.select %pred, %rhs, %lhs : i1, f32
    dataflow.graph.return %ctrl, %selected : none, f32
  }

  dataflow.graph.func private @integer_mix(%ctrl: none) -> (none, i32) {
    %wide = dataflow.constant %ctrl {const_value = 305419896 : i64} : i64
    %value = llvm.trunc %wide : i64 to i32
    %amount = dataflow.constant %ctrl {const_value = 4 : i32} : i32
    %rotated = llvm.intr.fshl(%value, %value, %amount) : (i32, i32, i32) -> i32
    %mask = dataflow.constant %ctrl {const_value = 255 : i32} : i32
    %xored = arith.xori %rotated, %mask : i32
    %modulus = dataflow.constant %ctrl {const_value = 13 : i32} : i32
    %reduced = arith.remui %xored, %modulus : i32
    %offset = dataflow.constant %ctrl {const_value = 5 : i32} : i32
    %subtracted = arith.subi %reduced, %offset : i32
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %is_nonzero = arith.cmpi ne, %subtracted, %zero : i32
    %fallback = dataflow.constant %ctrl {const_value = 99 : i32} : i32
    %selected = arith.select %is_nonzero, %subtracted, %fallback : i32
    dataflow.graph.return %ctrl, %selected : none, i32
  }

  dataflow.graph.func private @byte_swap(%ctrl: none) -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 305419896 : i32} : i32
    %swapped = llvm.intr.bswap(%value) : (i32) -> i32
    dataflow.graph.return %ctrl, %swapped : none, i32
  }

  dataflow.graph.func private @zext_bits(%ctrl: none) -> (none, i64) {
    %value = dataflow.constant %ctrl {const_value = -1 : i32} : i32
    %wide = llvm.zext %value : i32 to i64
    dataflow.graph.return %ctrl, %wide : none, i64
  }
}
