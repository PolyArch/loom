// RUN: loom-dfg-sim %s --graph compare_select --arg 0=none --output %t.compare.json
// RUN: FileCheck %s --check-prefix=COMPARE < %t.compare.json
// RUN: loom-dfg-sim %s --graph integer_mix --arg 0=none --output %t.integer.json
// RUN: FileCheck %s --check-prefix=INTEGER < %t.integer.json

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
}
