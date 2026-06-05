// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/mean LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/mean/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l1 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l1/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l2 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l2/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/reduction LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/reduction/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/prefix_sum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/prefix_sum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/cumsum LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/cumsum/dfg_check.sh
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/vecadd/main_func.dfg.mlir %t.dir/vecadd.dfg.report.json %t.dir/vecadd.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh mean %t.dir/mean/main_func.dfg.mlir %t.dir/mean.dfg.report.json %t.dir/mean.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecnorm_l1 %t.dir/vecnorm_l1/main_func.dfg.mlir %t.dir/vecnorm_l1.dfg.report.json %t.dir/vecnorm_l1.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecnorm_l2 %t.dir/vecnorm_l2/main_func.dfg.mlir %t.dir/vecnorm_l2.dfg.report.json %t.dir/vecnorm_l2.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh reduction %t.dir/reduction/main_func.dfg.mlir %t.dir/reduction.dfg.report.json %t.dir/reduction.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh prefix_sum %t.dir/prefix_sum/main_func.dfg.mlir %t.dir/prefix_sum.dfg.report.json %t.dir/prefix_sum.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh cumsum %t.dir/cumsum/main_func.dfg.mlir %t.dir/cumsum.dfg.report.json %t.dir/cumsum.dfg.summary.csv
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/vecadd.mapping.csv --artifact %t.dir/vecadd.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/mean/main_func.dfg.mlir --graph g_t_mean_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mean --output %t.dir/mean.mapping.csv --artifact %t.dir/mean.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l1/main_func.dfg.mlir --graph g_t_vecnorm_l1_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecnorm_l1 --output %t.dir/vecnorm_l1.mapping.csv --artifact %t.dir/vecnorm_l1.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l2/main_func.dfg.mlir --graph g_t_vecnorm_l2_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecnorm_l2 --output %t.dir/vecnorm_l2.mapping.csv --artifact %t.dir/vecnorm_l2.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/reduction/main_func.dfg.mlir --graph g_t_reduce_sum_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload reduction --output %t.dir/reduction.mapping.csv --artifact %t.dir/reduction.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/prefix_sum/main_func.dfg.mlir --graph g_t_prefix_sum_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload prefix_sum --output %t.dir/prefix_sum.mapping.csv --artifact %t.dir/prefix_sum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/cumsum/main_func.dfg.mlir --graph g_t_cumsum_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload cumsum --output %t.dir/cumsum.mapping.csv --artifact %t.dir/cumsum.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecadd.dfg.report.json --mapping-artifact %t.dir/vecadd.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecadd.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/mean.dfg.report.json --mapping-artifact %t.dir/mean.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/mean.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecnorm_l1.dfg.report.json --mapping-artifact %t.dir/vecnorm_l1.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecnorm_l1.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecnorm_l2.dfg.report.json --mapping-artifact %t.dir/vecnorm_l2.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecnorm_l2.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/reduction.dfg.report.json --mapping-artifact %t.dir/reduction.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/reduction.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/prefix_sum.dfg.report.json --mapping-artifact %t.dir/prefix_sum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/prefix_sum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/cumsum.dfg.report.json --mapping-artifact %t.dir/cumsum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/cumsum.cgra.report.json
// RUN: bash %S/../app/run_sim_cycle_summary.sh --dfg-report %t.dir/vecadd.dfg.report.json --cgra-report %t.dir/vecadd.cgra.report.json --dfg-report %t.dir/mean.dfg.report.json --cgra-report %t.dir/mean.cgra.report.json --dfg-report %t.dir/vecnorm_l1.dfg.report.json --cgra-report %t.dir/vecnorm_l1.cgra.report.json --dfg-report %t.dir/vecnorm_l2.dfg.report.json --cgra-report %t.dir/vecnorm_l2.cgra.report.json --dfg-report %t.dir/reduction.dfg.report.json --cgra-report %t.dir/reduction.cgra.report.json --dfg-report %t.dir/prefix_sum.dfg.report.json --cgra-report %t.dir/prefix_sum.cgra.report.json --dfg-report %t.dir/cumsum.dfg.report.json --cgra-report %t.dir/cumsum.cgra.report.json --output %t.dir/summary.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-NEXT: vecadd,643,653,pass
// SUMMARY-NEXT: mean,643,653,pass
// SUMMARY-NEXT: vecnorm_l1,643,654,pass
// SUMMARY-NEXT: vecnorm_l2,771,783,pass
// SUMMARY-NEXT: reduction,579,589,pass
// SUMMARY-NEXT: prefix_sum,835,852,pass
// SUMMARY-NEXT: cumsum,14339,14356,pass
