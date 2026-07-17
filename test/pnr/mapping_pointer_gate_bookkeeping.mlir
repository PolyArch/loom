// RUN: not loom-pnr-map --dfg-mlir %s --graph pointer_gate_memory_rejected --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload pointer_gate_memory_rejected --output %t.rejected.csv --artifact %t.rejected.json 2>&1 | FileCheck %s --check-prefix=REJECTED
// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph projected_carry --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload projected_carry --output %t.projected.csv --artifact %t.projected.json
// RUN: FileCheck %s --check-prefix=PROJECTED-CSV < %t.projected.csv
// RUN: FileCheck %s --check-prefix=PROJECTED-JSON < %t.projected.json

// REJECTED: finalized graph routes memory capability through 'dataflow.gate'

// PROJECTED-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PROJECTED-CSV-NEXT: projected_carry,shared_reduction_adg,projected_carry__projected_carry__shared_reduction_adg,5,4,0,0,pass,mapped software graph to fabric resources

// PROJECTED-JSON-DAG: "status": "pass"
// PROJECTED-JSON-DAG: "placed_records": 5
// PROJECTED-JSON-DAG: "unrouted_edges": 0
// PROJECTED-JSON-DAG: "operation": "dataflow.carry"
// PROJECTED-JSON-DAG: "operation": "dataflow.gate"
// PROJECTED-JSON-DAG: "operation": "dataflow.demux"
// PROJECTED-JSON-DAG: "operation": "dataflow.sync"
// PROJECTED-JSON-DAG: "edge_ref": "dataflow.carry#0.result0->dataflow.gate#0.operand1"
// PROJECTED-JSON-DAG: "edge_ref": "dataflow.carry#0.result0->dataflow.demux#0.operand1"

module {
  dataflow.graph.func private @pointer_gate_memory_rejected(
      %ctrl: none, %cond: i1, %ptr: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 0, 1, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %after_cond, %after_ptr = dataflow.gate %cond, %ptr : !llvm.ptr
    dataflow.graph.return values() streams() memories()
        complete(%ctrl : none)
  }

  dataflow.graph.func private @projected_carry(
      %ctrl: none, %phase: i1, %init: i32, %next: i32, %unit: none)
      -> (none, i32, i32)
      attributes {input_segments = array<i32: 0, 4, 0>,
                  result_segments = array<i32: 1, 1, 0>} {
    %raw = dataflow.carry %phase, %init, %next : i32
    %body_phase, %body = dataflow.gate %phase, %raw : i32
    %exit:2 = dataflow.demux %phase, %raw : (i1, i32) -> (i32, i32)
    %closed:2 = dataflow.demux %phase, %unit
        : (i1, none) -> (none, none)
    %retired:2 = dataflow.sync %closed#0, %exit#0
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32)
        streams(%body : i32) memories() complete(%retired#0 : none)
  }
}
