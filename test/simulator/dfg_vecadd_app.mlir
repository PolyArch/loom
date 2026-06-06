// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/main_func.dfg.mlir %t.report.json %t.summary.csv
// RUN: FileCheck %s --check-prefix=REPORT < %t.report.json
// RUN: FileCheck %s --check-prefix=REDUCTION < %t.reduction.report.json
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.summary.csv

// REPORT-DAG: "kind": "dfg_sim_report"
// REPORT-DAG: "workload": "vecadd"
// REPORT-DAG: "graph": "g_t_vecadd_0_0"
// REPORT-DAG: "status": "pass"
// REPORT-DAG: "metric_definition": "optimistic_operation_latency_sum"
// REPORT-DAG: "optimistic_cycles": 960
// REPORT-DAG: "wavefront_steps": 67
// REPORT-DAG: "event_count": 320
// REPORT-DAG: "none"

// REDUCTION-DAG: "kind": "dfg_sim_report"
// REDUCTION-DAG: "workload": "vecadd"
// REDUCTION-DAG: "graph": "g_t_main_red_0_0"
// REDUCTION-DAG: "status": "pass"
// REDUCTION-DAG: "optimistic_cycles": 643
// REDUCTION-DAG: "f32:3024"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY: vecadd,1603,,blocked,DFG-sim report available
