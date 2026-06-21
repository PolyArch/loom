// RUN: loom-pnr-map --dfg-mlir %s --graph integer_trunc_to_store --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload integer_trunc_to_store --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: integer_trunc_to_store,shared_reduction_adg,integer_trunc_to_store__integer_trunc_to_store__shared_reduction_adg,3,3,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "llvm.trunc"
// JSON-DAG: "resource_kind": "fabric.op"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.trunc#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.store#0.operand3"
// JSON-DAG: "edge_ref": "llvm.trunc#0.result0->dataflow.store#0.operand2"
// JSON-NOT: "missing hardware resource for software op llvm.trunc"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph integer_extend_trunc_to_store --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload integer_extend_trunc_to_store --output %t.extend.mapping.csv --artifact %t.extend.mapping.json
// RUN: FileCheck %s --check-prefix=EXTEND-CSV < %t.extend.mapping.csv
// RUN: FileCheck %s --check-prefix=EXTEND-JSON < %t.extend.mapping.json

// EXTEND-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// EXTEND-CSV-NEXT: integer_extend_trunc_to_store,shared_reduction_adg,integer_extend_trunc_to_store__integer_extend_trunc_to_store__shared_reduction_adg,4,4,0,0,pass

// EXTEND-JSON-DAG: "status": "pass"
// EXTEND-JSON-DAG: "operation": "llvm.sext"
// EXTEND-JSON-DAG: "operation": "llvm.trunc"
// EXTEND-JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.sext#0.operand0"
// EXTEND-JSON-DAG: "edge_ref": "llvm.sext#0.result0->llvm.trunc#0.operand0"
// EXTEND-JSON-DAG: "edge_ref": "llvm.trunc#0.result0->dataflow.store#0.operand2"
// EXTEND-JSON-NOT: "missing hardware resource for software op llvm.sext"
// EXTEND-JSON-NOT: ".out"
// EXTEND-JSON-NOT: ".in"

module {
  dataflow.graph.func private @integer_trunc_to_store(
      %ctrl: none, %input: memref<?xi32>, %output: memref<?xi16>,
      %idx: index) -> none {
    %data, %done = dataflow.load %input[%idx] %ctrl : memref<?xi32>
    %narrow = llvm.trunc %data : i32 to i16
    %stored = dataflow.store %output[%idx] %narrow %done : memref<?xi16>
    dataflow.graph.return %stored : none
  }

  dataflow.graph.func private @integer_extend_trunc_to_store(
      %ctrl: none, %input: memref<?xi16>, %output: memref<?xi16>,
      %idx: index) -> none {
    %data, %done = dataflow.load %input[%idx] %ctrl : memref<?xi16>
    %wide = llvm.sext %data : i16 to i32
    %narrow = llvm.trunc %wide : i32 to i16
    %stored = dataflow.store %output[%idx] %narrow %done : memref<?xi16>
    dataflow.graph.return %stored : none
  }
}
