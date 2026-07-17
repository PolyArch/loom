// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph shared_index_carry --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload shared_index_carry --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: shared_index_carry,shared_reduction_adg,shared_index_carry__shared_index_carry__shared_reduction_adg,10,13,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "shared_index_carry"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 10
// JSON-DAG: "routed_edges": 13
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.carry#1.operand2"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand2"
// JSON-DAG: "edge_ref": "dataflow.carry#1.result0->arith.addi#0.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand0"
// JSON-DAG: "edge_ref": "dataflow.carry#1.result0->dataflow.load#0.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::mem.load#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.constant#0.result0->dataflow.carry#1.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand1"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @shared_index_carry(%ctrl: none, %end: i32,
                                                  %start: i32, %step: i32,
                                                  %zero_f: f32,
                                                  %mem: memref<?xf32>)
      -> (none, f32) {
    %index, %rwc = dataflow.stream %end, %start, %step
        step add while slt : i32
    %sum_carried = dataflow.carry %rwc, %zero_f, %sum : f32
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %stride = dataflow.invariant %rwc, %one : i32
    %idx_carried = dataflow.carry %rwc, %zero, %next_idx : i32
    %next_idx = arith.addi %idx_carried, %stride : i32
    %load_idx = arith.index_cast %idx_carried : i32 to index
    %data, %done = dataflow.load %mem[%load_idx] %ctrl : memref<?xf32>
    %sum = arith.addf %sum_carried, %data : f32
    %synced = dataflow.sync %done : (none) -> none
    dataflow.graph.return %synced, %sum_carried : none, f32
  }
}
