// RUN: loom-pnr-map --dfg-mlir %s --graph pointer_gate_value_only --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload pointer_gate_value_only --output %t.value.csv --artifact %t.value.json
// RUN: FileCheck %s --check-prefix=VALUE-CSV < %t.value.csv
// RUN: FileCheck %s --check-prefix=VALUE-JSON < %t.value.json
// RUN: loom-pnr-map --dfg-mlir %s --graph pointer_gate_cond_used --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload pointer_gate_cond_used --output %t.cond.csv --artifact %t.cond.json
// RUN: FileCheck %s --check-prefix=COND-CSV < %t.cond.csv
// RUN: FileCheck %s --check-prefix=COND-JSON < %t.cond.json

// VALUE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// VALUE-CSV-NEXT: pointer_gate_value_only,shared_reduction_adg,pointer_gate_value_only__pointer_gate_value_only__shared_reduction_adg,2,2,0,0,pass,mapped software graph to fabric resources

// VALUE-JSON-DAG: "status": "pass"
// VALUE-JSON-DAG: "placed_records": 2
// VALUE-JSON-DAG: "unplaced_records": 0
// VALUE-JSON-NOT: "operation": "dataflow.gate"

// COND-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// COND-CSV-NEXT: pointer_gate_cond_used,shared_reduction_adg,pointer_gate_cond_used__pointer_gate_cond_used__shared_reduction_adg,3,2,1,0,fail,unrouted software edges lack Fabric ADG connectivity

// COND-JSON-DAG: "status": "fail"
// COND-JSON-DAG: "placed_records": 3
// COND-JSON-DAG: "unplaced_records": 0
// COND-JSON-DAG: "unrouted_edges": 1
// COND-JSON-DAG: "operation": "dataflow.gate"
// COND-JSON-DAG: "edge_ref": "dataflow.gate#0.result0->dataflow.carry#0.operand0"

module {
  dataflow.graph.func private @pointer_gate_value_only(
      %ctrl: none, %cond: i1, %ptr: !llvm.ptr, %zero: i32, %one: i32)
      -> (none, i32, !llvm.ptr) {
    %after_cond, %after_ptr = dataflow.gate %cond, %ptr : !llvm.ptr
    %carried = dataflow.carry %cond, %zero, %next : i32
    %next = arith.addi %carried, %one : i32
    dataflow.graph.return %ctrl, %carried, %after_ptr : none, i32, !llvm.ptr
  }

  dataflow.graph.func private @pointer_gate_cond_used(
      %ctrl: none, %cond: i1, %ptr: !llvm.ptr, %zero: i32, %one: i32)
      -> (none, i32) {
    %after_cond, %after_ptr = dataflow.gate %cond, %ptr : !llvm.ptr
    %carried = dataflow.carry %after_cond, %zero, %next : i32
    %next = arith.addi %carried, %one : i32
    dataflow.graph.return %ctrl, %carried : none, i32
  }
}
