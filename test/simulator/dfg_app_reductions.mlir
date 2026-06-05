// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/dotproduct LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/dotproduct/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l2 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l2/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/integrate_trapz LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/integrate_trapz/dfg_check.sh
// RUN: mkdir -p %t.dir/reports
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/vecadd/main_func.dfg.mlir %t.dir/reports/vecadd.report.json %t.dir/summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecsum %t.dir/vecsum/main_func.dfg.mlir %t.dir/reports/vecsum.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh dotproduct %t.dir/dotproduct/main_func.dfg.mlir %t.dir/reports/dotproduct.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecnorm_l2 %t.dir/vecnorm_l2/main_func.dfg.mlir %t.dir/reports/vecnorm_l2.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh integrate_trapz %t.dir/integrate_trapz/main_func.dfg.mlir %t.dir/reports/integrate_trapz.report.json %t.dir/summary.csv --append
// RUN: FileCheck %s --check-prefix=VECADD < %t.dir/reports/vecadd.report.json
// RUN: FileCheck %s --check-prefix=VECSUM < %t.dir/reports/vecsum.report.json
// RUN: FileCheck %s --check-prefix=DOTPRODUCT < %t.dir/reports/dotproduct.report.json
// RUN: FileCheck %s --check-prefix=VECNORM-L2 < %t.dir/reports/vecnorm_l2.report.json
// RUN: FileCheck %s --check-prefix=INTEGRATE-TRAPZ < %t.dir/reports/integrate_trapz.report.json
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

// VECNORM-L2-DAG: "kind": "dfg_sim_report"
// VECNORM-L2-DAG: "workload": "vecnorm_l2"
// VECNORM-L2-DAG: "graph": "g_t_vecnorm_l2_red_0_0"
// VECNORM-L2-DAG: "status": "pass"
// VECNORM-L2-DAG: "optimistic_cycles": 132
// VECNORM-L2-DAG: "event_count": 517
// VECNORM-L2-DAG: "i32:619"

// INTEGRATE-TRAPZ-DAG: "kind": "dfg_sim_report"
// INTEGRATE-TRAPZ-DAG: "workload": "integrate_trapz"
// INTEGRATE-TRAPZ-DAG: "graph": "g_t_integrate_trapz_red_0_0"
// INTEGRATE-TRAPZ-DAG: "status": "pass"
// INTEGRATE-TRAPZ-DAG: "optimistic_cycles": 22
// INTEGRATE-TRAPZ-DAG: "event_count": 169
// INTEGRATE-TRAPZ-DAG: "f32:0.335938"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-DAG: dotproduct,131,,blocked,DFG-sim report available
// SUMMARY-DAG: integrate_trapz,22,,blocked,DFG-sim report available
// SUMMARY-DAG: vecadd,131,,blocked,DFG-sim report available
// SUMMARY-DAG: vecnorm_l2,132,,blocked,DFG-sim report available
// SUMMARY-DAG: vecsum,131,,blocked,DFG-sim report available
