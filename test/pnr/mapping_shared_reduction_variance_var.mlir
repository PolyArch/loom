// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/variance LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/variance/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/variance/main_func.dfg.mlir --graph g_t_variance_red_1_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload variance --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: variance,shared_reduction_adg,variance__g_t_variance_red_1_0__shared_reduction_adg,9,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "variance"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "mapping_id": "variance__g_t_variance_red_1_0__shared_reduction_adg"
// JSON-DAG: "placed_records": 9
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "status": "pass"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.subf#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.invariant#1.result0->arith.subf#0.operand1"
// JSON-DAG: "edge_ref": "arith.subf#0.result0->llvm.intr.fmuladd#0.operand0"
// JSON-DAG: "edge_ref": "arith.subf#0.result0->llvm.intr.fmuladd#0.operand1"
// JSON-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.invariant#1.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::mem.load#0.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#10.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#10.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#9.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#4.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#11.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#11.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#9.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#5.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#12.operand2"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#12.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#20.operand0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#13.operand2"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#13.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#20.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#0.result4"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#7.operand0"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
