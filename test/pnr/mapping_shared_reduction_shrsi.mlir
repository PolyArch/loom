// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph signed_shift --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload signed_shift --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: signed_shift,shared_reduction_adg,signed_shift__signed_shift__shared_reduction_adg,2,1,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "arith.shrsi"
// JSON-DAG: "operation": "dataflow.sync"
// JSON-DAG: "edge_ref": "arith.shrsi#0.result0->dataflow.sync#0.operand1"
// JSON-DAG: "hardware": "shared_reduction_adg::fabric.op#
// JSON-NOT: "resource_kind=fabric.op operation=arith.shrsi

module {
  dataflow.graph.func private @signed_shift(%ctrl: none, %value: i32, %amount: i32)
      -> (none, i32) {
    %shifted = arith.shrsi %value, %amount : i32
    dataflow.graph.return %ctrl, %shifted : none, i32
  }
}
