// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/main_func.dfg.mlir %t.report.json %t.summary.csv
// RUN: FileCheck %s --check-prefix=REPORT < %t.report.json
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.summary.csv

// REPORT-DAG: "kind": "dfg_sim_report"
// REPORT-DAG: "workload": "vecadd"
// REPORT-DAG: "graph": "g_t_main_red_0_0"
// REPORT-DAG: "status": "pass"
// REPORT-DAG: "metric_definition": "optimistic_event_count"
// REPORT-DAG: "optimistic_cycles": 387
// REPORT-DAG: "wavefront_steps": 131
// REPORT-DAG: "event_count": 387
// REPORT-DAG: "f32:3024"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY: vecadd,387,,blocked,DFG-sim report available
