// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/shared-memory-reduction.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph spmm_computed_address --hardware-mlir %t.dir/shared-memory-reduction.mlir --hardware shared_memory_reduction_adg --workload spmm_computed_address --output %t.csv --artifact %t.json
// RUN: FileCheck %s --check-prefix=CSV < %t.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: spmm_computed_address,shared_memory_reduction_adg,spmm_computed_address__spmm_computed_address__shared_memory_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "software": "dataflow.constant#0"
// JSON-DAG: "software": "arith.index_cast#0"
// JSON-DAG: "edge_ref": "dataflow.constant#0.result0->arith.index_cast#0.operand0"
// JSON-DAG: "edge_ref": "arith.index_cast#0.result0->arith.addi#0.operand1"
// JSON-DAG: "segment_kind": "buffer"
// JSON-DAG: "hardware_ref": "shared_memory_reduction_adg::fabric.switch#0"
// JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.store#0.operand1"
// JSON-DAG: "sink_endpoint": "shared_memory_reduction_adg::mem.load#0.operand0"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unrouted_edge_details": []
// JSON-DAG: "status": "pass"

module {
  dataflow.graph private @spmm_computed_address(
      %ctrl: none, %row: i64, %input: memref<?xi32>,
      %output: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 1, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %bias_value = dataflow.constant %ctrl {const_value = 1 : i64} : i64
    %row_index = arith.index_cast %row : i64 to index
    %bias_index = arith.index_cast %bias_value : i64 to index
    %address = arith.addi %row_index, %bias_index : index
    %data, %loaded = dataflow.load %input[%address] %ctrl : memref<?xi32>
    %stored = dataflow.store %output[%address] %data %loaded : memref<?xi32>
    dataflow.graph.return values() streams() memories()
        complete(%stored : none)
  }
}
