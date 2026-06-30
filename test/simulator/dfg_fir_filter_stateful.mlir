// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/fir_filter_stateful LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/fir_filter_stateful/dfg_check.sh
// RUN: mkdir -p %t.dir/reports
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh fir_filter_stateful %t.dir/fir_filter_stateful/main_func.dfg.mlir %t.dir/reports/fir_filter_stateful.report.json %t.dir/summary.csv
// RUN: FileCheck %s --check-prefix=REPORT < %t.dir/reports/fir_filter_stateful.report.json
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// REPORT: "diagnostics": []
// REPORT: "dynamic_work_items": 4
// REPORT: "arg4": [
// REPORT-NEXT: "f32:0.250000",
// REPORT-NEXT: "f32:-0.125000",
// REPORT-NEXT: "f32:0.500000",
// REPORT-NEXT: "f32:0.375000",
// REPORT-NEXT: "f32:-0.250000"
// REPORT-NEXT: ]
// REPORT: "arg6": [
// REPORT-NEXT: "f32:4",
// REPORT-NEXT: "f32:3",
// REPORT-NEXT: "f32:2",
// REPORT-NEXT: "f32:1"
// REPORT-NEXT: ]
// REPORT: "final_outputs": [
// REPORT-NEXT: "none",
// REPORT-NEXT: "f32:1.250000"
// REPORT-NEXT: ]
// REPORT: "graph": "g_t_fir_filter_stateful_kernel_red_0_0"
// REPORT: "kind": "dfg_sim_report"
// REPORT: "dataflow.load": 8
// REPORT: "llvm.intr.fmuladd": 4
// REPORT: "optimistic_cycles": 126
// REPORT: "status": "pass"
// REPORT: "workload": "fir_filter_stateful"

// SUMMARY: fir_filter_stateful,126,,blocked,DFG-sim report available; CGRA-sim requires Fabric ADG and mapping artifact evidence
