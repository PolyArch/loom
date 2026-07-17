// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph cmsis_vector_sum_s8 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload cmsis_vector_sum_s8 --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: cmsis_vector_sum_s8,shared_reduction_adg,cmsis_vector_sum_s8__cmsis_vector_sum_s8__shared_reduction_adg,{{[1-9][0-9]*}},{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "cmsis_vector_sum_s8"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.addi#1.operand0"
// JSON-DAG: "edge_ref": "arith.addi#1.result0->arith.muli#0.operand0"
// JSON-DAG: "edge_ref": "arith.muli#0.result0->arith.addi#2.operand1"
// JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.addi#2.operand0"
// JSON-DAG: "edge_ref": "arith.addi#2.result0->dataflow.store#0.operand2"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

module {
  dataflow.graph.func private @cmsis_vector_sum_s8(
      %ctrl: none, %src_index: index, %dst_index: index, %acc: i32,
      %bias: i32, %scale: i32, %src: memref<?xi8>, %dst: memref<?xi32>)
      -> none
      attributes {input_segments = array<i32: 5, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %loaded_i8, %src_done = dataflow.load %src[%src_index] %ctrl
        : memref<?xi8>
    %loaded_i32 = llvm.sext %loaded_i8 : i8 to i32
    %sum = arith.addi %acc, %loaded_i32 : i32
    %biased = arith.addi %sum, %bias : i32
    %scaled = arith.muli %biased, %scale : i32
    %old, %dst_done = dataflow.load %dst[%dst_index] %ctrl
        : memref<?xi32>
    %updated = arith.addi %old, %scaled : i32
    %stored = dataflow.store %dst[%dst_index] %updated %ctrl
        : memref<?xi32>
    dataflow.graph.return values() streams() memories()
        complete(%stored : none)
  }
}
