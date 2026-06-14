// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: vecsum,shared_reduction_adg,vecsum__g_t_vecsum_red_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "vecsum"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 5
// JSON-DAG: "routed_edges": 6
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unplaced_records": 0
// JSON-DAG: "config_records": 97
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "segment_kind": "module_path"
// JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.carry#0.operand2"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#2.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#15.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#15.operand0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#15.result0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#15.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#1.operand2"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addi#0.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::mem.load#0.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#7.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#7.operand0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#7.result0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#7.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#2.operand0"
// JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.sync#0.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::mem.load#0.result1"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand0"
// JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#0.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::mem.load#0.operand0"
// JSON-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.carry#0.operand0"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#0.result1"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#1.operand0"
// JSON-DAG: "register": "segment_count"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
