// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph shared_index_carry --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload shared_index_carry --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: shared_index_carry,shared_reduction_adg,shared_index_carry__shared_index_carry__shared_reduction_adg,13,19,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "shared_index_carry"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 13
// JSON-DAG: "routed_edges": 19
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
// JSON-DAG: "edge_ref": "dataflow.demux#1.result0->dataflow.sync#1.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand1"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph private @shared_index_carry(%ctrl: none, %end: i32,
                                                  %start: i32, %step: i32,
                                                  %zero_f: f32,
                                                  %unit: none,
                                                  %mem: memref<?xf32>)
      -> (f32)
      attributes {input_segments = array<i32: 4, 1, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
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
    %closed:2 = dataflow.demux %rwc, %unit
        : (i1, none) -> (none, none)
    %final:2 = dataflow.demux %rwc, %sum_carried
        : (i1, f32) -> (f32, f32)
    %effects:2 = dataflow.sync %closed#0, %done
        : (none, none) -> (none, none)
    %retired:2 = dataflow.sync %effects#0, %final#0
        : (none, f32) -> (none, f32)
    dataflow.graph.return values(%retired#1 : f32) streams() memories()
        complete(%retired#0 : none)
  }
}
