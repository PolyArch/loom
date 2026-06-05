// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.dir/mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.dir/mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,status,diagnostic
// CSV-NEXT: vecsum,shared_reduction_adg,vecsum__shared_reduction_adg,6,8,0,pass

// JSON-DAG: "kind": "pnr_mapping"
// JSON-DAG: "workload": "vecsum"
// JSON-DAG: "hardware": "shared_reduction_adg"
// JSON-DAG: "status": "pass"
// JSON-DAG: "placed_records": 6
// JSON-DAG: "routed_edges": 8
// JSON-DAG: "config_records": 43
// JSON-DAG: "config_bitstream"
// JSON-DAG: "register": "sw_configs.step_op"
// JSON-DAG: "value": "+="
// JSON-DAG: "register": "sw_configs.bitmask"
// JSON-DAG: "register": "from_software_id"
