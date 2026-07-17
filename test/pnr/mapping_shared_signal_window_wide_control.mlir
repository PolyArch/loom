// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-signal-window --output %t.dir/shared-signal-window.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph signal_window_wide_control --hardware-mlir %t.dir/shared-signal-window.mlir --hardware shared_signal_window_adg --workload signal_window_wide_control --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: signal_window_wide_control,shared_signal_window_adg,signal_window_wide_control__signal_window_wide_control__shared_signal_window_adg,{{[1-9][0-9]*}},{{[1-9][0-9]*}},0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "operation": "arith.addi"
// JSON-DAG: "operation": "arith.cmpi"
// JSON-DAG: "operation": "arith.index_cast"
// JSON-DAG: "operation": "arith.trunci"
// JSON-DAG: "operation": "dataflow.load"
// JSON-DAG: "operation": "dataflow.store"
// JSON-NOT: "resource_pressure"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @signal_window_wide_control(
      %ctrl: none,
      %lhs: i64,
      %rhs: i64,
      %input: memref<?xi32>,
      %output: memref<?xi32>) -> none
      attributes {input_segments = array<i32: 2, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : i64
    %idx = arith.index_cast %sum : i64 to index
    %narrow = arith.trunci %sum : i64 to i32
    %lt = arith.cmpi ult, %lhs, %rhs : i64
    %eq = arith.cmpi eq, %sum, %rhs : i64
    %loaded, %load_done = dataflow.load %input[%idx] %ctrl : memref<?xi32>
    %selected = arith.select %lt, %loaded, %narrow : i32
    %value = arith.select %eq, %selected, %loaded : i32
    %store_done = dataflow.store %output[%idx] %value %load_done : memref<?xi32>
    %done:2 = dataflow.sync %load_done, %store_done : (none, none) -> (none, none)
    dataflow.graph.return %done#0 : none
  }
}
