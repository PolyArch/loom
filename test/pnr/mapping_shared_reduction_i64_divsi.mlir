// RUN: loom-pnr-map --dfg-mlir %s --graph i64_signed_div --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload i64_signed_div --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: i64_signed_div,shared_reduction_adg,i64_signed_div__i64_signed_div__shared_reduction_adg,1,0,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "arith.divsi"
// JSON-DAG: "hardware": "shared_reduction_adg::fabric.op#
// JSON-NOT: "resource_kind=fabric.op operation=arith.divsi

module {
  dataflow.graph.func private @i64_signed_div(%ctrl: none, %lhs: i64, %rhs: i64)
      -> (none, i64) {
    %quotient = arith.divsi %lhs, %rhs : i64
    dataflow.graph.return %ctrl, %quotient : none, i64
  }
}
