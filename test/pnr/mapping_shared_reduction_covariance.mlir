// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/covariance LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/covariance/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/covariance/main_func.dfg.mlir --graph g_t_covariance_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload covariance --output %t.dir/sums.mapping.csv --artifact %t.dir/sums.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/covariance/main_func.dfg.mlir --graph g_t_covariance_kernel_red_1_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload covariance --output %t.dir/cov.mapping.csv --artifact %t.dir/cov.mapping.json
// RUN: FileCheck %s --check-prefix=SUMS-CSV < %t.dir/sums.mapping.csv
// RUN: FileCheck %s --check-prefix=SUMS-JSON < %t.dir/sums.mapping.json
// RUN: FileCheck %s --check-prefix=COV-CSV < %t.dir/cov.mapping.csv
// RUN: FileCheck %s --check-prefix=COV-JSON < %t.dir/cov.mapping.json

// SUMS-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SUMS-CSV-NEXT: covariance,shared_reduction_adg,covariance__g_t_covariance_kernel_red_0_0__shared_reduction_adg,8,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// SUMS-JSON-DAG: "kind": "pnr_mapping"
// SUMS-JSON-DAG: "workload": "covariance"
// SUMS-JSON-DAG: "hardware": "shared_reduction_adg"
// SUMS-JSON-DAG: "mapping_id": "covariance__g_t_covariance_kernel_red_0_0__shared_reduction_adg"
// SUMS-JSON-DAG: "placed_records": 8
// SUMS-JSON-DAG: "unrouted_edges": 0
// SUMS-JSON-DAG: "unplaced_records": 0
// SUMS-JSON-DAG: "status": "pass"
// SUMS-JSON-DAG: "operation": "arith.addf"
// SUMS-JSON-DAG: "software": "arith.addf#0"
// SUMS-JSON-DAG: "operation": "arith.addf"
// SUMS-JSON-DAG: "software": "arith.addf#1"
// SUMS-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand1"
// SUMS-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.addf#1.operand1"
// SUMS-JSON-NOT: ".out"
// SUMS-JSON-NOT: ".in"

// COV-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// COV-CSV-NEXT: covariance,shared_reduction_adg,covariance__g_t_covariance_kernel_red_1_0__shared_reduction_adg,10,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// COV-JSON-DAG: "kind": "pnr_mapping"
// COV-JSON-DAG: "workload": "covariance"
// COV-JSON-DAG: "hardware": "shared_reduction_adg"
// COV-JSON-DAG: "mapping_id": "covariance__g_t_covariance_kernel_red_1_0__shared_reduction_adg"
// COV-JSON-DAG: "placed_records": 10
// COV-JSON-DAG: "unrouted_edges": 0
// COV-JSON-DAG: "unplaced_records": 0
// COV-JSON-DAG: "status": "pass"
// COV-JSON-DAG: "operation": "arith.subf"
// COV-JSON-DAG: "software": "arith.subf#0"
// COV-JSON-DAG: "operation": "arith.subf"
// COV-JSON-DAG: "software": "arith.subf#1"
// COV-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.subf#0.operand0"
// COV-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.subf#1.operand0"
// COV-JSON-NOT: ".out"
// COV-JSON-NOT: ".in"
