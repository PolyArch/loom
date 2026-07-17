// RUN: loom-pnr-map --dfg-mlir %s --graph control_token_branch_merge --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload control_token_branch_merge --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: control_token_branch_merge,shared_reduction_adg,control_token_branch_merge__control_token_branch_merge__shared_reduction_adg,4,4,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "dataflow.demux"
// JSON-DAG: "operation": "dataflow.mux"
// JSON-DAG: "operation": "dataflow.store"
// JSON-DAG: "edge_ref": "dataflow.demux#0.result0->dataflow.mux#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.demux#0.result1->dataflow.store#0.operand3"
// JSON-DAG: "edge_ref": "dataflow.demux#1.result1->dataflow.store#0.operand2"
// JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.mux#0.operand2"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph private @control_token_branch_merge(
      %ctrl: none, %sel: i1, %idx: index, %value: i32,
      %mem: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ctrl_false, %ctrl_true =
        dataflow.demux %sel, %ctrl : (i1, none) -> (none, none)
    %data_false, %data_true =
        dataflow.demux %sel, %value : (i1, i32) -> (i32, i32)
    %stored = dataflow.store %mem[%idx] %data_true %ctrl_true : memref<?xi32>
    %done = dataflow.mux %sel, %ctrl_false, %stored : (i1, none, none) -> none
    dataflow.graph.return %done : none
  }
}
