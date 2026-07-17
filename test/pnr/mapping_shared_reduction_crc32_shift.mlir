// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph crc32_shift_mix --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload crc32_shift_mix --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: crc32_shift_mix,shared_reduction_adg,crc32_shift_mix__crc32_shift_mix__shared_reduction_adg,10,11,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "crc32_shift_mix"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "arith.shrui#0.result0->arith.xori#1.operand0"
// JSON-DAG: "edge_ref": "arith.shrui#1.result0->arith.xori#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.xori#1.operand1"
// JSON-DAG: "edge_ref": "arith.xori#0.result0->arith.andi#0.operand0"
// JSON-DAG: "edge_ref": "arith.andi#0.result0->dataflow.load#1.operand1"
// JSON-DAG: "edge_ref": "dataflow.sync#0.result0->dataflow.sync#1.operand0"
// JSON-DAG: "edge_ref": "arith.xori#1.result0->dataflow.sync#1.operand1"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @crc32_shift_mix(
      %ctrl: none, %index: index, %carry: i32, %inner: i32,
      %byte_shift: i32, %mask: index, %input: memref<?xi32>,
      %table: memref<?xi32>) -> (none, i32)
      attributes {input_segments = array<i32: 5, 0, 2>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data, %input_done = dataflow.load %input[%index] %ctrl : memref<?xi32>
    %carry_shifted = arith.shrui %carry, %byte_shift : i32
    %bit_shift = arith.shli %inner, %byte_shift : i32
    %data_shifted = arith.shrui %data, %bit_shift : i32
    %mixed = arith.xori %data_shifted, %carry : i32
    %mixed_index = arith.index_cast %mixed : i32 to index
    %table_index = arith.andi %mixed_index, %mask : index
    %table_data, %table_done = dataflow.load %table[%table_index] %ctrl
        : memref<?xi32>
    %next = arith.xori %carry_shifted, %table_data : i32
    dataflow.graph.return %ctrl, %next : none, i32
  }
}
