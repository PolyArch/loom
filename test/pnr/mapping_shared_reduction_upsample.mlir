// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/upsample LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/upsample/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/upsample/main_func.dfg.mlir --graph g_t_upsample_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload upsample --output %t.dir/upsample.mapping.csv --artifact %t.dir/upsample.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/upsample.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/upsample.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: upsample,shared_reduction_adg,upsample__g_t_upsample_0_0__shared_reduction_adg,6,6,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "upsample"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 6
// JSON-DAG: "routed_edges": 6
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "edge_ref": "arith.shrui#0.result0->dataflow.store#0.operand1"
// JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.result0"
// JSON-DAG: "sink_endpoint": "shared_reduction_adg::mem.store#0.operand0"
// JSON-DAG: "segment_kind": "module_path"
// JSON-NOT: ".out"
// JSON-NOT: ".in"
