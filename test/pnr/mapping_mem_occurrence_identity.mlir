// RUN: loom-pnr-map --dfg-mlir %s --graph mixed_mem_resources --hardware-mlir %S/mapping_mem_occurrence_identity.hardware.mlir.in --hardware named_dual_mem_adg --workload named_dual_mem --output %t.named.csv --artifact %t.named.json
// RUN: FileCheck %s --check-prefix=NAMED < %t.named.json
// RUN: loom-mapping-estimate --mapping-artifact %t.named.json --hardware-mlir %S/mapping_mem_occurrence_identity.hardware.mlir.in --output %t.named.estimate.json
// RUN: FileCheck %s --check-prefix=NAMED-ESTIMATE < %t.named.estimate.json
// RUN: loom-pnr-map --dfg-mlir %s --graph mixed_mem_resources --hardware-mlir %S/mapping_mem_occurrence_identity.hardware.mlir.in --hardware direct_dual_mem_adg --workload direct_dual_mem --output %t.direct.csv --artifact %t.direct.json
// RUN: FileCheck %s --check-prefix=DIRECT < %t.direct.json
// RUN: loom-mapping-estimate --mapping-artifact %t.direct.json --hardware-mlir %S/mapping_mem_occurrence_identity.hardware.mlir.in --output %t.direct.estimate.json
// RUN: FileCheck %s --check-prefix=DIRECT-ESTIMATE < %t.direct.estimate.json

// NAMED-DAG: "hardware": "named_dual_mem_adg"
// NAMED-DAG: "hardware": "named_dual_mem_adg::mem.load#0"
// NAMED-DAG: "hardware": "named_dual_mem_adg::mem.load#1"
// NAMED-DAG: "hardware": "named_dual_mem_adg::mem.load#2"
// NAMED-DAG: "hardware": "named_dual_mem_adg::mem.load#3"
// NAMED-DAG: "hardware": "named_dual_mem_adg::mem.store#0"
// NAMED-DAG: "hardware": "named_dual_mem_adg::mem.store#1"
// NAMED-DAG: "source_endpoint": "named_dual_mem_adg::mem.load#0.result0"
// NAMED-DAG: "source_endpoint": "named_dual_mem_adg::mem.load#1.result0"
// NAMED-DAG: "source_endpoint": "named_dual_mem_adg::mem.load#2.result0"
// NAMED-DAG: "placed_records": 10
// NAMED-DAG: "routed_edges": 12
// NAMED-DAG: "status": "pass"
// NAMED-DAG: "unplaced_records": 0
// NAMED-DAG: "unrouted_edges": 0

// DIRECT-DAG: "hardware": "direct_dual_mem_adg"
// DIRECT-DAG: "hardware": "direct_dual_mem_adg::mem.load#0"
// DIRECT-DAG: "hardware": "direct_dual_mem_adg::mem.load#1"
// DIRECT-DAG: "hardware": "direct_dual_mem_adg::mem.load#2"
// DIRECT-DAG: "hardware": "direct_dual_mem_adg::mem.load#3"
// DIRECT-DAG: "hardware": "direct_dual_mem_adg::mem.store#0"
// DIRECT-DAG: "hardware": "direct_dual_mem_adg::mem.store#1"
// DIRECT-DAG: "source_endpoint": "direct_dual_mem_adg::mem.load#0.result0"
// DIRECT-DAG: "source_endpoint": "direct_dual_mem_adg::mem.load#1.result0"
// DIRECT-DAG: "source_endpoint": "direct_dual_mem_adg::mem.load#2.result0"
// DIRECT-DAG: "placed_records": 10
// DIRECT-DAG: "routed_edges": 12
// DIRECT-DAG: "status": "pass"
// DIRECT-DAG: "unplaced_records": 0
// DIRECT-DAG: "unrouted_edges": 0

// NAMED-ESTIMATE-DAG: "kind": "mapping_estimate_report"
// NAMED-ESTIMATE-DAG: "config_records": 150
// NAMED-ESTIMATE-DAG: "placed_records": 10
// NAMED-ESTIMATE-DAG: "routed_edges": 12
// NAMED-ESTIMATE-DAG: "status": "pass"
// DIRECT-ESTIMATE-DAG: "kind": "mapping_estimate_report"
// DIRECT-ESTIMATE-DAG: "config_records": 150
// DIRECT-ESTIMATE-DAG: "placed_records": 10
// DIRECT-ESTIMATE-DAG: "routed_edges": 12
// DIRECT-ESTIMATE-DAG: "status": "pass"

module {
  dataflow.graph.func private @mixed_mem_resources(
      %ctrl: none, %index: index, %store0_value: i32, %store1_value: i32,
      %load0_mem: memref<?xi32>, %load1_mem: memref<?xi32>,
      %load2_mem: memref<?xi32>, %load3_mem: memref<?xi32>,
      %store0_mem: memref<?xi32>, %store1_mem: memref<?xi32>)
      -> (none, i32)
      attributes {input_segments = array<i32: 3, 0, 6>,
                  result_segments = array<i32: 1, 0, 0>} {
    %load0, %load0_done =
        dataflow.load %load0_mem[%index] %ctrl : memref<?xi32>
    %load1, %load1_done =
        dataflow.load %load1_mem[%index] %ctrl : memref<?xi32>
    %load2, %load2_done =
        dataflow.load %load2_mem[%index] %ctrl : memref<?xi32>
    %load3, %load3_done =
        dataflow.load %load3_mem[%index] %ctrl : memref<?xi32>
    %sum0 = arith.addi %load0, %load1 : i32
    %sum1 = arith.addi %sum0, %load2 : i32
    %store0_done = dataflow.store %store0_mem[%index]
        %store0_value %ctrl : memref<?xi32>
    %store1_done = dataflow.store %store1_mem[%index]
        %store1_value %ctrl : memref<?xi32>
    %effects:6 = dataflow.sync %load0_done, %load1_done, %load2_done,
        %load3_done, %store0_done, %store1_done
        : (none, none, none, none, none, none)
          -> (none, none, none, none, none, none)
    %published:2 = dataflow.sync %effects#0, %sum1
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
}
