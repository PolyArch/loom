// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph string_hash_power --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload string_hash_power --output %t.power.mapping.csv --artifact %t.power.mapping.json
// RUN: FileCheck %s --check-prefix=POWER-CSV < %t.power.mapping.csv
// RUN: FileCheck %s --check-prefix=POWER-JSON < %t.power.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph string_hash_window --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload string_hash_window --output %t.window.mapping.csv --artifact %t.window.mapping.json
// RUN: FileCheck %s --check-prefix=WINDOW-CSV < %t.window.mapping.csv
// RUN: FileCheck %s --check-prefix=WINDOW-JSON < %t.window.mapping.json

// POWER-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// POWER-CSV-NEXT: string_hash_power,shared_reduction_adg,string_hash_power__string_hash_power__shared_reduction_adg,6,{{[0-9]+}},0,0,pass,mapped software graph to fabric resources
// POWER-JSON-DAG: "status": "pass"
// POWER-JSON-DAG: "operation": "arith.shli"
// POWER-JSON-DAG: "operation": "arith.remui"
// POWER-JSON-DAG: "edge_ref": "arith.shli#0.result0->arith.remui#0.operand0"
// POWER-JSON-DAG: "edge_ref": "arith.remui#0.result0->dataflow.carry#0.operand2"
// POWER-JSON-DAG: "unrouted_edges": 0
// POWER-JSON-NOT: ".out"
// POWER-JSON-NOT: ".in"

// WINDOW-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// WINDOW-CSV-NEXT: string_hash_window,shared_reduction_adg,string_hash_window__string_hash_window__shared_reduction_adg,8,7,1,1,fail,missing hardware resource for software op dataflow.stream
// WINDOW-JSON-DAG: "status": "fail"
// WINDOW-JSON-DAG: "unplaced_records": 1
// WINDOW-JSON-DAG: "operation": "dataflow.load"
// WINDOW-JSON-DAG: "operation": "arith.addi"
// WINDOW-JSON-DAG: "operation": "arith.shli"
// WINDOW-JSON-DAG: "operation": "arith.remui"
// WINDOW-JSON-DAG: "edge_ref": "arith.shli#0.result0->arith.addi#0.operand1"
// WINDOW-JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.remui#0.operand0"
// WINDOW-JSON-DAG: "edge_ref": "arith.remui#0.result0->dataflow.carry#0.operand2"
// WINDOW-JSON-DAG: "unrouted_edges": 1
// WINDOW-JSON-DAG: "missing hardware resource for software op dataflow.stream
// WINDOW-JSON-NOT: ".out"
// WINDOW-JSON-NOT: ".in"

module {
  dataflow.graph.func private @string_hash_power(%ctrl: none, %start: i32,
                                                  %end: i32, %step: i32,
                                                  %shift: i32, %modulus: i32,
                                                  %init: i32)
      -> (none, i32) {
    %index, %rwc = dataflow.stream %start, %end, %step
        step add while slt : i32
    %stable_modulus = dataflow.invariant %rwc, %modulus : i32
    %stable_shift = dataflow.invariant %rwc, %shift : i32
    %carried = dataflow.carry %rwc, %init, %next : i32
    %shifted = arith.shli %carried, %stable_shift : i32
    %next = arith.remui %shifted, %stable_modulus : i32
    dataflow.graph.return %ctrl, %carried : none, i32
  }

  dataflow.graph.func private @string_hash_window(
      %ctrl: none, %start: i64, %end: i64, %step: i64, %shift: i32,
      %input: memref<?xi32>, %modulus: i32, %init: i32) -> (none, i32) {
    %index, %rwc = dataflow.stream %start, %end, %step
        step add while slt : i64
    %stable_modulus = dataflow.invariant %rwc, %modulus : i32
    %stable_shift = dataflow.invariant %rwc, %shift : i32
    %carried = dataflow.carry %rwc, %init, %next : i32
    %shifted = arith.shli %carried, %stable_shift : i32
    %load_idx = arith.index_cast %index : i64 to index
    %data, %done = dataflow.load %input[%load_idx] %ctrl : memref<?xi32>
    %sum = arith.addi %data, %shifted : i32
    %next = arith.remui %sum, %stable_modulus : i32
    %synced = dataflow.sync %done : (none) -> none
    dataflow.graph.return %synced, %carried : none, i32
  }
}
