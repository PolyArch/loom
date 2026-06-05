// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/dotproduct LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/dotproduct/dfg_check.sh
// RUN: mkdir -p %t.dir/reports
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/vecadd/main_func.dfg.mlir %t.dir/reports/vecadd.report.json %t.dir/summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecsum %t.dir/vecsum/main_func.dfg.mlir %t.dir/reports/vecsum.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh dotproduct %t.dir/dotproduct/main_func.dfg.mlir %t.dir/reports/dotproduct.report.json %t.dir/summary.csv --append
// RUN: FileCheck %s --check-prefix=VECADD < %t.dir/reports/vecadd.report.json
// RUN: FileCheck %s --check-prefix=VECSUM < %t.dir/reports/vecsum.report.json
// RUN: FileCheck %s --check-prefix=DOTPRODUCT < %t.dir/reports/dotproduct.report.json
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// VECADD-DAG: "kind": "dfg_sim_report"
// VECADD-DAG: "workload": "vecadd"
// VECADD-DAG: "graph": "g_t_main_red_0_0"
// VECADD-DAG: "status": "pass"
// VECADD-DAG: "optimistic_cycles": 131
// VECADD-DAG: "event_count": 453
// VECADD-DAG: "f32:3024"

// VECSUM-DAG: "kind": "dfg_sim_report"
// VECSUM-DAG: "workload": "vecsum"
// VECSUM-DAG: "graph": "g_t_vecsum_red_0_0"
// VECSUM-DAG: "status": "pass"
// VECSUM-DAG: "optimistic_cycles": 131
// VECSUM-DAG: "event_count": 453
// VECSUM-DAG: "i32:2116"

// DOTPRODUCT-DAG: "kind": "dfg_sim_report"
// DOTPRODUCT-DAG: "workload": "dotproduct"
// DOTPRODUCT-DAG: "graph": "g_t_dotproduct_red_0_0"
// DOTPRODUCT-DAG: "status": "pass"
// DOTPRODUCT-DAG: "optimistic_cycles": 131
// DOTPRODUCT-DAG: "event_count": 517
// DOTPRODUCT-DAG: "f32:2016"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-DAG: dotproduct,131,,blocked,DFG-sim report available
// SUMMARY-DAG: vecadd,131,,blocked,DFG-sim report available
// SUMMARY-DAG: vecsum,131,,blocked,DFG-sim report available
