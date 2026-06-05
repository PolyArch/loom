// RUN: rm -rf %t.dir
// RUN: BUILD_DIR=%t.dir bash %S/../app/vecadd/dfg_check.sh
// RUN: bash %S/run_vecadd_dfg_sim.sh %t.dir/main_func.dfg.mlir %t.report.json %t.summary.csv
// RUN: FileCheck %s --check-prefix=REPORT < %t.report.json
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.summary.csv

// REPORT-DAG: "kind": "dfg_sim_report"
// REPORT-DAG: "workload": "vecadd"
// REPORT-DAG: "graph": "g_t_main_red_0_0"
// REPORT-DAG: "status": "pass"
// REPORT-DAG: "optimistic_cycles": 131
// REPORT-DAG: "event_count": 453
// REPORT-DAG: "f32:3024"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY: vecadd,131,,blocked,DFG-sim report available
