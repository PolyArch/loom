// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/bit_reverse LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/bit_reverse/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/conv1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/conv1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/convolve_1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/convolve_1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/correlation LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/correlation/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/cumsum LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/cumsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/reduction LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/reduction/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/spmv LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/spmv/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/mean LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/mean/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/dotproduct LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/dotproduct/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l1 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l1/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l2 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l2/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/prefix_sum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/prefix_sum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/integrate_trapz LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/integrate_trapz/dfg_check.sh
// RUN: mkdir -p %t.dir/reports
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh bit_reverse %t.dir/bit_reverse/main_func.dfg.mlir %t.dir/reports/bit_reverse.report.json %t.dir/summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh conv1d %t.dir/conv1d/main_func.dfg.mlir %t.dir/reports/conv1d.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh convolve_1d %t.dir/convolve_1d/main_func.dfg.mlir %t.dir/reports/convolve_1d.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh correlation %t.dir/correlation/main_func.dfg.mlir %t.dir/reports/correlation.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh cumsum %t.dir/cumsum/main_func.dfg.mlir %t.dir/reports/cumsum.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/vecadd/main_func.dfg.mlir %t.dir/reports/vecadd.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecsum %t.dir/vecsum/main_func.dfg.mlir %t.dir/reports/vecsum.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh reduction %t.dir/reduction/main_func.dfg.mlir %t.dir/reports/reduction.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh spmv %t.dir/spmv/main_func.dfg.mlir %t.dir/reports/spmv.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh mean %t.dir/mean/main_func.dfg.mlir %t.dir/reports/mean.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh dotproduct %t.dir/dotproduct/main_func.dfg.mlir %t.dir/reports/dotproduct.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecnorm_l1 %t.dir/vecnorm_l1/main_func.dfg.mlir %t.dir/reports/vecnorm_l1.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecnorm_l2 %t.dir/vecnorm_l2/main_func.dfg.mlir %t.dir/reports/vecnorm_l2.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh prefix_sum %t.dir/prefix_sum/main_func.dfg.mlir %t.dir/reports/prefix_sum.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh integrate_trapz %t.dir/integrate_trapz/main_func.dfg.mlir %t.dir/reports/integrate_trapz.report.json %t.dir/summary.csv --append
// RUN: FileCheck %s --check-prefix=BIT-REVERSE < %t.dir/reports/bit_reverse.report.json
// RUN: FileCheck %s --check-prefix=CONV1D < %t.dir/reports/conv1d.report.json
// RUN: FileCheck %s --check-prefix=CONVOLVE-1D < %t.dir/reports/convolve_1d.report.json
// RUN: FileCheck %s --check-prefix=CORRELATION < %t.dir/reports/correlation.report.json
// RUN: FileCheck %s --check-prefix=CUMSUM < %t.dir/reports/cumsum.report.json
// RUN: FileCheck %s --check-prefix=VECADD < %t.dir/reports/vecadd.report.json
// RUN: FileCheck %s --check-prefix=VECSUM < %t.dir/reports/vecsum.report.json
// RUN: FileCheck %s --check-prefix=REDUCTION < %t.dir/reports/reduction.report.json
// RUN: FileCheck %s --check-prefix=SPMV < %t.dir/reports/spmv.report.json
// RUN: FileCheck %s --check-prefix=MEAN < %t.dir/reports/mean.report.json
// RUN: FileCheck %s --check-prefix=DOTPRODUCT < %t.dir/reports/dotproduct.report.json
// RUN: FileCheck %s --check-prefix=VECNORM-L1 < %t.dir/reports/vecnorm_l1.report.json
// RUN: FileCheck %s --check-prefix=VECNORM-L2 < %t.dir/reports/vecnorm_l2.report.json
// RUN: FileCheck %s --check-prefix=PREFIX-SUM < %t.dir/reports/prefix_sum.report.json
// RUN: FileCheck %s --check-prefix=INTEGRATE-TRAPZ < %t.dir/reports/integrate_trapz.report.json
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// BIT-REVERSE-DAG: "kind": "dfg_sim_report"
// BIT-REVERSE-DAG: "workload": "bit_reverse"
// BIT-REVERSE-DAG: "graph": "g_t_bit_reverse_kernel_0_0"
// BIT-REVERSE-DAG: "status": "pass"
// BIT-REVERSE-DAG: "optimistic_cycles": 267
// BIT-REVERSE-DAG: "event_count": 267
// BIT-REVERSE-DAG: "i32:510274632"
// BIT-REVERSE-DAG: "i32:0"

// CONV1D-DAG: "kind": "dfg_sim_report"
// CONV1D-DAG: "workload": "conv1d"
// CONV1D-DAG: "graph": "g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0"
// CONV1D-DAG: "status": "pass"
// CONV1D-DAG: "optimistic_cycles": 38
// CONV1D-DAG: "wavefront_steps": 13
// CONV1D-DAG: "event_count": 38
// CONV1D-DAG: "f32:5"

// CONVOLVE-1D-DAG: "kind": "dfg_sim_report"
// CONVOLVE-1D-DAG: "workload": "convolve_1d"
// CONVOLVE-1D-DAG: "graph": "g_t_convolve_1d_kernel_0_0"
// CONVOLVE-1D-DAG: "status": "pass"
// CONVOLVE-1D-DAG: "optimistic_cycles": 94
// CONVOLVE-1D-DAG: "wavefront_steps": 19
// CONVOLVE-1D-DAG: "event_count": 94
// CONVOLVE-1D-DAG: "f32:1.000000"

// CORRELATION-DAG: "kind": "dfg_sim_report"
// CORRELATION-DAG: "workload": "correlation"
// CORRELATION-DAG: "graph": "g_t_correlation_kernel_0_0"
// CORRELATION-DAG: "status": "pass"
// CORRELATION-DAG: "optimistic_cycles": 202
// CORRELATION-DAG: "wavefront_steps": 37
// CORRELATION-DAG: "event_count": 202
// CORRELATION-DAG: "f32:16"

// CUMSUM-DAG: "kind": "dfg_sim_report"
// CUMSUM-DAG: "workload": "cumsum"
// CUMSUM-DAG: "graph": "g_t_cumsum_kernel_red_0_0"
// CUMSUM-DAG: "status": "pass"
// CUMSUM-DAG: "optimistic_cycles": 7171
// CUMSUM-DAG: "wavefront_steps": 2052
// CUMSUM-DAG: "event_count": 7171
// CUMSUM-DAG: "f32:5620"

// VECADD-DAG: "kind": "dfg_sim_report"
// VECADD-DAG: "workload": "vecadd"
// VECADD-DAG: "graph": "g_t_main_red_0_0"
// VECADD-DAG: "status": "pass"
// VECADD-DAG: "metric_definition": "optimistic_event_count"
// VECADD-DAG: "optimistic_cycles": 387
// VECADD-DAG: "wavefront_steps": 131
// VECADD-DAG: "event_count": 387
// VECADD-DAG: "f32:3024"

// VECSUM-DAG: "kind": "dfg_sim_report"
// VECSUM-DAG: "workload": "vecsum"
// VECSUM-DAG: "graph": "g_t_vecsum_red_0_0"
// VECSUM-DAG: "status": "pass"
// VECSUM-DAG: "metric_definition": "optimistic_event_count"
// VECSUM-DAG: "optimistic_cycles": 387
// VECSUM-DAG: "wavefront_steps": 131
// VECSUM-DAG: "event_count": 387
// VECSUM-DAG: "i32:2116"

// REDUCTION-DAG: "kind": "dfg_sim_report"
// REDUCTION-DAG: "workload": "reduction"
// REDUCTION-DAG: "graph": "g_t_reduce_sum_red_0_0"
// REDUCTION-DAG: "status": "pass"
// REDUCTION-DAG: "optimistic_cycles": 387
// REDUCTION-DAG: "event_count": 387
// REDUCTION-DAG: "i32:2016"

// SPMV-DAG: "kind": "dfg_sim_report"
// SPMV-DAG: "workload": "spmv"
// SPMV-DAG: "graph": "g_t_spmv_kernel_red_0_0"
// SPMV-DAG: "status": "pass"
// SPMV-DAG: "optimistic_cycles": 25
// SPMV-DAG: "wavefront_steps": 11
// SPMV-DAG: "event_count": 25
// SPMV-DAG: "i32:12"

// MEAN-DAG: "kind": "dfg_sim_report"
// MEAN-DAG: "workload": "mean"
// MEAN-DAG: "graph": "g_t_mean_kernel_red_0_0"
// MEAN-DAG: "status": "pass"
// MEAN-DAG: "optimistic_cycles": 387
// MEAN-DAG: "event_count": 387
// MEAN-DAG: "f32:2016"

// DOTPRODUCT-DAG: "kind": "dfg_sim_report"
// DOTPRODUCT-DAG: "workload": "dotproduct"
// DOTPRODUCT-DAG: "graph": "g_t_dotproduct_red_0_0"
// DOTPRODUCT-DAG: "status": "pass"
// DOTPRODUCT-DAG: "metric_definition": "optimistic_event_count"
// DOTPRODUCT-DAG: "optimistic_cycles": 451
// DOTPRODUCT-DAG: "wavefront_steps": 131
// DOTPRODUCT-DAG: "event_count": 451
// DOTPRODUCT-DAG: "f32:2016"

// VECNORM-L1-DAG: "kind": "dfg_sim_report"
// VECNORM-L1-DAG: "workload": "vecnorm_l1"
// VECNORM-L1-DAG: "graph": "g_t_vecnorm_l1_red_0_0"
// VECNORM-L1-DAG: "status": "pass"
// VECNORM-L1-DAG: "optimistic_cycles": 451
// VECNORM-L1-DAG: "event_count": 451
// VECNORM-L1-DAG: "i32:171"

// VECNORM-L2-DAG: "kind": "dfg_sim_report"
// VECNORM-L2-DAG: "workload": "vecnorm_l2"
// VECNORM-L2-DAG: "graph": "g_t_vecnorm_l2_red_0_0"
// VECNORM-L2-DAG: "status": "pass"
// VECNORM-L2-DAG: "metric_definition": "optimistic_event_count"
// VECNORM-L2-DAG: "optimistic_cycles": 451
// VECNORM-L2-DAG: "wavefront_steps": 132
// VECNORM-L2-DAG: "event_count": 451
// VECNORM-L2-DAG: "i32:619"

// PREFIX-SUM-DAG: "kind": "dfg_sim_report"
// PREFIX-SUM-DAG: "workload": "prefix_sum"
// PREFIX-SUM-DAG: "graph": "g_t_prefix_sum_red_0_0"
// PREFIX-SUM-DAG: "status": "pass"
// PREFIX-SUM-DAG: "optimistic_cycles": 451
// PREFIX-SUM-DAG: "event_count": 451
// PREFIX-SUM-DAG: "i32:2016"

// INTEGRATE-TRAPZ-DAG: "kind": "dfg_sim_report"
// INTEGRATE-TRAPZ-DAG: "workload": "integrate_trapz"
// INTEGRATE-TRAPZ-DAG: "graph": "g_t_integrate_trapz_red_0_0"
// INTEGRATE-TRAPZ-DAG: "status": "pass"
// INTEGRATE-TRAPZ-DAG: "metric_definition": "optimistic_event_count"
// INTEGRATE-TRAPZ-DAG: "optimistic_cycles": 147
// INTEGRATE-TRAPZ-DAG: "wavefront_steps": 21
// INTEGRATE-TRAPZ-DAG: "event_count": 147
// INTEGRATE-TRAPZ-DAG: "f32:0.335938"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-DAG: bit_reverse,267,,blocked,DFG-sim report available
// SUMMARY-DAG: conv1d,38,,blocked,DFG-sim report available
// SUMMARY-DAG: convolve_1d,94,,blocked,DFG-sim report available
// SUMMARY-DAG: correlation,202,,blocked,DFG-sim report available
// SUMMARY-DAG: cumsum,7171,,blocked,DFG-sim report available
// SUMMARY-DAG: dotproduct,451,,blocked,DFG-sim report available
// SUMMARY-DAG: integrate_trapz,147,,blocked,DFG-sim report available
// SUMMARY-DAG: mean,387,,blocked,DFG-sim report available
// SUMMARY-DAG: prefix_sum,451,,blocked,DFG-sim report available
// SUMMARY-DAG: reduction,387,,blocked,DFG-sim report available
// SUMMARY-DAG: spmv,25,,blocked,DFG-sim report available
// SUMMARY-DAG: vecadd,387,,blocked,DFG-sim report available
// SUMMARY-DAG: vecnorm_l1,451,,blocked,DFG-sim report available
// SUMMARY-DAG: vecnorm_l2,451,,blocked,DFG-sim report available
// SUMMARY-DAG: vecsum,387,,blocked,DFG-sim report available
