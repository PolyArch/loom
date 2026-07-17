// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph ctlz_map --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload ctlz_map --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: ctlz_map,shared_reduction_adg,ctlz_map__ctlz_map__shared_reduction_adg,2,1,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "llvm.intr.ctlz"
// JSON-DAG: "operation": "dataflow.sync"
// JSON-DAG: "edge_ref": "llvm.intr.ctlz#0.result0->dataflow.sync#0.operand1"
// JSON-DAG: "hardware": "shared_reduction_adg::fabric.op#
// JSON-NOT: "unsupported PnR graph operation: llvm.intr.ctlz"

module {
  dataflow.graph private @ctlz_map(%ctrl: none, %value: i32)
      -> (i32) {
    %zeros = "llvm.intr.ctlz"(%value) <{is_zero_poison = false}> : (i32) -> i32
    dataflow.graph.return %ctrl, %zeros : none, i32
  }
}
