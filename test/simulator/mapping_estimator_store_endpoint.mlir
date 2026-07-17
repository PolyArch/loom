// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %S/../pnr/mapping_mem_route.mlir --graph cfft_red3_fmul_pair --hardware-mlir %t.dir/hardware.mlir --hardware shared_memory_reduction_adg --workload cfft_red3_fmul_pair --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.dir/mapping.json --hardware-mlir %t.dir/hardware.mlir --output %t.dir/estimate.json
// RUN: FileCheck %s --check-prefix=ESTIMATE < %t.dir/estimate.json

// MAPPING-DAG: "status": "pass"
// MAPPING-DAG: "operation": "dataflow.store"
// MAPPING-DAG: "resource_kind": "fabric.mem.store"
// MAPPING-DAG: "edge_ref": "arith.mulf#1.result0->dataflow.store#1.operand2"
// MAPPING-DAG: "sink_endpoint": "shared_memory_reduction_adg::mem.store#1.operand1"

// ESTIMATE-DAG: "kind": "mapping_estimate_report"
// ESTIMATE-DAG: "workload": "cfft_red3_fmul_pair"
// ESTIMATE-DAG: "status": "pass"
// ESTIMATE-DAG: "mapping_id": "cfft_red3_fmul_pair__cfft_red3_fmul_pair__shared_memory_reduction_adg"
// ESTIMATE-DAG: "store_address_score": 4
