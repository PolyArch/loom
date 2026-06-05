// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecsum %t.dir/vecsum/main_func.dfg.mlir %t.dir/vecsum.dfg.report.json %t.dir/dfg-summary.csv
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecsum.dfg.report.json --mapping-artifact %t.dir/mapping.json --output %t.dir/vecsum.cgra.report.json
// RUN: FileCheck %s --check-prefix=CGRA < %t.dir/vecsum.cgra.report.json
// RUN: loom-sim-cycle-summary --dfg-report %t.dir/vecsum.dfg.report.json --cgra-report %t.dir/vecsum.cgra.report.json --output %t.dir/summary.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// MAPPING-DAG: "schedule": "spatial"
// MAPPING-DAG: "resource_kind": "fabric.mem.load"
// MAPPING-DAG: "config_records": 43
// MAPPING-DAG: "config_bitstream"

// CGRA-DAG: "kind": "cgra_sim_report"
// CGRA-DAG: "workload": "vecsum"
// CGRA-DAG: "mapping_id": "vecsum__shared_reduction_adg"
// CGRA-DAG: "status": "pass"
// CGRA-DAG: "dfg_cycles": 131
// CGRA-DAG: "route_latency_cycles": 8
// CGRA-DAG: "memory_latency_cycles": 4
// CGRA-DAG: "temporal_penalty_cycles": 0
// CGRA-DAG: "hardware_aware_cycles": 143
// CGRA-DAG: "config_records": 43

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-NEXT: vecsum,131,143,pass
