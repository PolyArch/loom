// RUN: loom-pnr-map --dfg-mlir %s --graph i64_modmul --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload i64_modmul --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: i64_modmul,shared_reduction_adg,i64_modmul__i64_modmul__shared_reduction_adg,9,10,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "llvm.zext"
// JSON-DAG: "operation": "arith.muli"
// JSON-DAG: "operation": "arith.remui"
// JSON-DAG: "operation": "llvm.trunc"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.zext#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->llvm.zext#1.operand0"
// JSON-DAG: "edge_ref": "llvm.zext#0.result0->arith.muli#0.operand1"
// JSON-DAG: "edge_ref": "llvm.zext#1.result0->arith.muli#0.operand0"
// JSON-DAG: "edge_ref": "arith.muli#0.result0->arith.remui#0.operand0"
// JSON-DAG: "edge_ref": "arith.remui#0.result0->llvm.trunc#0.operand0"
// JSON-DAG: "edge_ref": "llvm.trunc#0.result0->dataflow.store#0.operand2"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @i64_modmul(
      %ctrl: none, %modulus: i64, %idx: index,
      %input_a: memref<?xi32>, %input_b: memref<?xi32>,
      %output: memref<?xi32>) -> none
      attributes {input_segments = array<i32: 2, 0, 3>,
                  result_segments = array<i32: 0, 0, 0>} {
    %a, %a_done = dataflow.load %input_a[%idx] %ctrl : memref<?xi32>
    %a64 = llvm.zext %a : i32 to i64
    %b, %b_done = dataflow.load %input_b[%idx] %ctrl : memref<?xi32>
    %b64 = llvm.zext %b : i32 to i64
    %product = arith.muli %b64, %a64 : i64
    %remainder = arith.remui %product, %modulus : i64
    %narrow = llvm.trunc %remainder overflow<nuw> : i64 to i32
    %stored = dataflow.store %output[%idx] %narrow %ctrl : memref<?xi32>
    %done:3 = dataflow.sync %a_done, %b_done, %stored
        : (none, none, none) -> (none, none, none)
    dataflow.graph.return %done#0 : none
  }
}
