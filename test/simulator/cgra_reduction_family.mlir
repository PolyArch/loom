// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/bit_reverse LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/bit_reverse/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/conv1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/conv1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/convolve_1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/convolve_1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/correlation LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/correlation/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/compare_swap LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/compare_swap/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/hash_mix LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/hash_mix/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/xor_block LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/xor_block/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/axpy LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/axpy/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/relu LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/relu/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/rotate_bits LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/rotate_bits/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/matvec LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/matvec/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecmul LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecmul/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecscale LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecscale/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/mean LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/mean/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l1 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l1/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l2 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l2/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/reduction LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/reduction/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/dotproduct LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/dotproduct/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/spmv LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/spmv/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/prefix_sum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/prefix_sum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/cumsum LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/cumsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/integrate_trapz LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/integrate_trapz/dfg_check.sh
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh bit_reverse %t.dir/bit_reverse/main_func.dfg.mlir %t.dir/bit_reverse.dfg.report.json %t.dir/bit_reverse.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh conv1d %t.dir/conv1d/main_func.dfg.mlir %t.dir/conv1d.dfg.report.json %t.dir/conv1d.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh convolve_1d %t.dir/convolve_1d/main_func.dfg.mlir %t.dir/convolve_1d.dfg.report.json %t.dir/convolve_1d.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh correlation %t.dir/correlation/main_func.dfg.mlir %t.dir/correlation.dfg.report.json %t.dir/correlation.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh compare_swap %t.dir/compare_swap/main_func.dfg.mlir %t.dir/compare_swap.dfg.report.json %t.dir/compare_swap.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh hash_mix %t.dir/hash_mix/main_func.dfg.mlir %t.dir/hash_mix.dfg.report.json %t.dir/hash_mix.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh xor_block %t.dir/xor_block/main_func.dfg.mlir %t.dir/xor_block.dfg.report.json %t.dir/xor_block.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh axpy %t.dir/axpy/main_func.dfg.mlir %t.dir/axpy.dfg.report.json %t.dir/axpy.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh relu %t.dir/relu/main_func.dfg.mlir %t.dir/relu.dfg.report.json %t.dir/relu.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh rotate_bits %t.dir/rotate_bits/main_func.dfg.mlir %t.dir/rotate_bits.dfg.report.json %t.dir/rotate_bits.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh matvec %t.dir/matvec/main_func.dfg.mlir %t.dir/matvec.dfg.report.json %t.dir/matvec.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/vecadd/main_func.dfg.mlir %t.dir/vecadd.dfg.report.json %t.dir/vecadd.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecmul %t.dir/vecmul/main_func.dfg.mlir %t.dir/vecmul.dfg.report.json %t.dir/vecmul.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecscale %t.dir/vecscale/main_func.dfg.mlir %t.dir/vecscale.dfg.report.json %t.dir/vecscale.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh mean %t.dir/mean/main_func.dfg.mlir %t.dir/mean.dfg.report.json %t.dir/mean.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecnorm_l1 %t.dir/vecnorm_l1/main_func.dfg.mlir %t.dir/vecnorm_l1.dfg.report.json %t.dir/vecnorm_l1.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecnorm_l2 %t.dir/vecnorm_l2/main_func.dfg.mlir %t.dir/vecnorm_l2.dfg.report.json %t.dir/vecnorm_l2.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh reduction %t.dir/reduction/main_func.dfg.mlir %t.dir/reduction.dfg.report.json %t.dir/reduction.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecsum %t.dir/vecsum/main_func.dfg.mlir %t.dir/vecsum.dfg.report.json %t.dir/vecsum.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh dotproduct %t.dir/dotproduct/main_func.dfg.mlir %t.dir/dotproduct.dfg.report.json %t.dir/dotproduct.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh spmv %t.dir/spmv/main_func.dfg.mlir %t.dir/spmv.dfg.report.json %t.dir/spmv.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh prefix_sum %t.dir/prefix_sum/main_func.dfg.mlir %t.dir/prefix_sum.dfg.report.json %t.dir/prefix_sum.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh cumsum %t.dir/cumsum/main_func.dfg.mlir %t.dir/cumsum.dfg.report.json %t.dir/cumsum.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh integrate_trapz %t.dir/integrate_trapz/main_func.dfg.mlir %t.dir/integrate_trapz.dfg.report.json %t.dir/integrate_trapz.dfg.summary.csv
// RUN: loom-pnr-map --dfg-mlir %t.dir/bit_reverse/main_func.dfg.mlir --graph g_t_bit_reverse_kernel_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload bit_reverse --output %t.dir/bit_reverse.mapping.csv --artifact %t.dir/bit_reverse.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/conv1d/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload conv1d --output %t.dir/conv1d.mapping.csv --artifact %t.dir/conv1d.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/convolve_1d/main_func.dfg.mlir --graph g_t_convolve_1d_kernel_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload convolve_1d --output %t.dir/convolve_1d.mapping.csv --artifact %t.dir/convolve_1d.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/correlation/main_func.dfg.mlir --graph g_t_correlation_kernel_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload correlation --output %t.dir/correlation.mapping.csv --artifact %t.dir/correlation.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/compare_swap/main_func.dfg.mlir --graph g_t_main_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload compare_swap --output %t.dir/compare_swap.mapping.csv --artifact %t.dir/compare_swap.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/hash_mix/main_func.dfg.mlir --graph g_t_main_1_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload hash_mix --output %t.dir/hash_mix.mapping.csv --artifact %t.dir/hash_mix.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/xor_block/main_func.dfg.mlir --graph g_t_xor_block_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload xor_block --output %t.dir/xor_block.mapping.csv --artifact %t.dir/xor_block.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/axpy/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_114axpy_candidateEPKjS1_Pjjj_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload axpy --output %t.dir/axpy.mapping.csv --artifact %t.dir/axpy.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/relu/main_func.dfg.mlir --graph g_t_relu_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload relu --output %t.dir/relu.core.mapping.csv --artifact %t.dir/relu.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/relu/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload relu --output %t.dir/relu.checksum.mapping.csv --artifact %t.dir/relu.checksum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/rotate_bits/main_func.dfg.mlir --graph g_t_rotate_bits_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload rotate_bits --output %t.dir/rotate_bits.mapping.csv --artifact %t.dir/rotate_bits.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_matvec_kernel_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.core.mapping.csv --artifact %t.dir/matvec.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.checksum.mapping.csv --artifact %t.dir/matvec.checksum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_vecadd_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/vecadd.core.mapping.csv --artifact %t.dir/vecadd.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/vecadd.reduction.mapping.csv --artifact %t.dir/vecadd.reduction.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecmul/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecmul --output %t.dir/vecmul.mapping.csv --artifact %t.dir/vecmul.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecscale/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_118vecscale_candidateEPKjjPjj_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecscale --output %t.dir/vecscale.mapping.csv --artifact %t.dir/vecscale.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/mean/main_func.dfg.mlir --graph g_t_mean_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mean --output %t.dir/mean.mapping.csv --artifact %t.dir/mean.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l1/main_func.dfg.mlir --graph g_t_vecnorm_l1_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecnorm_l1 --output %t.dir/vecnorm_l1.mapping.csv --artifact %t.dir/vecnorm_l1.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l2/main_func.dfg.mlir --graph g_t_vecnorm_l2_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecnorm_l2 --output %t.dir/vecnorm_l2.mapping.csv --artifact %t.dir/vecnorm_l2.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/reduction/main_func.dfg.mlir --graph g_t_reduce_sum_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload reduction --output %t.dir/reduction.mapping.csv --artifact %t.dir/reduction.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/vecsum.mapping.csv --artifact %t.dir/vecsum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/dotproduct/main_func.dfg.mlir --graph g_t_dotproduct_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload dotproduct --output %t.dir/dotproduct.mapping.csv --artifact %t.dir/dotproduct.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/spmv/main_func.dfg.mlir --graph g_t_spmv_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload spmv --output %t.dir/spmv.mapping.csv --artifact %t.dir/spmv.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/prefix_sum/main_func.dfg.mlir --graph g_t_prefix_sum_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload prefix_sum --output %t.dir/prefix_sum.mapping.csv --artifact %t.dir/prefix_sum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/cumsum/main_func.dfg.mlir --graph g_t_cumsum_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload cumsum --output %t.dir/cumsum.mapping.csv --artifact %t.dir/cumsum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/integrate_trapz/main_func.dfg.mlir --graph g_t_integrate_trapz_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload integrate_trapz --output %t.dir/integrate_trapz.mapping.csv --artifact %t.dir/integrate_trapz.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/bit_reverse.dfg.report.json --mapping-artifact %t.dir/bit_reverse.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/bit_reverse.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/conv1d.dfg.report.json --mapping-artifact %t.dir/conv1d.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/conv1d.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/convolve_1d.dfg.report.json --mapping-artifact %t.dir/convolve_1d.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/convolve_1d.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/correlation.dfg.report.json --mapping-artifact %t.dir/correlation.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/correlation.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/compare_swap.dfg.report.json --mapping-artifact %t.dir/compare_swap.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/compare_swap.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/hash_mix.dfg.report.json --mapping-artifact %t.dir/hash_mix.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/hash_mix.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/xor_block.dfg.report.json --mapping-artifact %t.dir/xor_block.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/xor_block.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/axpy.dfg.report.json --mapping-artifact %t.dir/axpy.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/axpy.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/relu.dfg.report.json --mapping-artifact %t.dir/relu.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/relu.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/relu.dfg.checksum.report.json --mapping-artifact %t.dir/relu.checksum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/relu.checksum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/rotate_bits.dfg.report.json --mapping-artifact %t.dir/rotate_bits.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/rotate_bits.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.row1.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.row1.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.row2.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.row2.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.row3.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.row3.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.checksum.report.json --mapping-artifact %t.dir/matvec.checksum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.checksum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecadd.dfg.report.json --mapping-artifact %t.dir/vecadd.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecadd.core.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecadd.dfg.reduction.report.json --mapping-artifact %t.dir/vecadd.reduction.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecadd.reduction.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecmul.dfg.report.json --mapping-artifact %t.dir/vecmul.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecmul.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecscale.dfg.report.json --mapping-artifact %t.dir/vecscale.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecscale.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/mean.dfg.report.json --mapping-artifact %t.dir/mean.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/mean.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecnorm_l1.dfg.report.json --mapping-artifact %t.dir/vecnorm_l1.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecnorm_l1.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecnorm_l2.dfg.report.json --mapping-artifact %t.dir/vecnorm_l2.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecnorm_l2.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/reduction.dfg.report.json --mapping-artifact %t.dir/reduction.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/reduction.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/vecsum.dfg.report.json --mapping-artifact %t.dir/vecsum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/vecsum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/dotproduct.dfg.report.json --mapping-artifact %t.dir/dotproduct.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/dotproduct.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/spmv.dfg.report.json --mapping-artifact %t.dir/spmv.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/spmv.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/prefix_sum.dfg.report.json --mapping-artifact %t.dir/prefix_sum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/prefix_sum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/cumsum.dfg.report.json --mapping-artifact %t.dir/cumsum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/cumsum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/integrate_trapz.dfg.report.json --mapping-artifact %t.dir/integrate_trapz.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/integrate_trapz.cgra.report.json
// RUN: bash %S/../app/run_sim_cycle_summary.sh --dfg-report %t.dir/axpy.dfg.report.json --cgra-report %t.dir/axpy.cgra.report.json --dfg-report %t.dir/bit_reverse.dfg.report.json --cgra-report %t.dir/bit_reverse.cgra.report.json --dfg-report %t.dir/conv1d.dfg.report.json --cgra-report %t.dir/conv1d.cgra.report.json --dfg-report %t.dir/convolve_1d.dfg.report.json --cgra-report %t.dir/convolve_1d.cgra.report.json --dfg-report %t.dir/correlation.dfg.report.json --cgra-report %t.dir/correlation.cgra.report.json --dfg-report %t.dir/compare_swap.dfg.report.json --cgra-report %t.dir/compare_swap.cgra.report.json --dfg-report %t.dir/hash_mix.dfg.report.json --cgra-report %t.dir/hash_mix.cgra.report.json --dfg-report %t.dir/xor_block.dfg.report.json --cgra-report %t.dir/xor_block.cgra.report.json --dfg-report %t.dir/relu.dfg.report.json --dfg-report %t.dir/relu.dfg.checksum.report.json --cgra-report %t.dir/relu.cgra.report.json --cgra-report %t.dir/relu.checksum.cgra.report.json --dfg-report %t.dir/rotate_bits.dfg.report.json --cgra-report %t.dir/rotate_bits.cgra.report.json --dfg-report %t.dir/matvec.dfg.report.json --dfg-report %t.dir/matvec.dfg.row1.report.json --dfg-report %t.dir/matvec.dfg.row2.report.json --dfg-report %t.dir/matvec.dfg.row3.report.json --dfg-report %t.dir/matvec.dfg.checksum.report.json --cgra-report %t.dir/matvec.cgra.report.json --cgra-report %t.dir/matvec.row1.cgra.report.json --cgra-report %t.dir/matvec.row2.cgra.report.json --cgra-report %t.dir/matvec.row3.cgra.report.json --cgra-report %t.dir/matvec.checksum.cgra.report.json --dfg-report %t.dir/vecadd.dfg.report.json --dfg-report %t.dir/vecadd.dfg.reduction.report.json --cgra-report %t.dir/vecadd.core.cgra.report.json --cgra-report %t.dir/vecadd.reduction.cgra.report.json --dfg-report %t.dir/vecmul.dfg.report.json --cgra-report %t.dir/vecmul.cgra.report.json --dfg-report %t.dir/vecscale.dfg.report.json --cgra-report %t.dir/vecscale.cgra.report.json --dfg-report %t.dir/mean.dfg.report.json --cgra-report %t.dir/mean.cgra.report.json --dfg-report %t.dir/vecnorm_l1.dfg.report.json --cgra-report %t.dir/vecnorm_l1.cgra.report.json --dfg-report %t.dir/vecnorm_l2.dfg.report.json --cgra-report %t.dir/vecnorm_l2.cgra.report.json --dfg-report %t.dir/reduction.dfg.report.json --cgra-report %t.dir/reduction.cgra.report.json --dfg-report %t.dir/vecsum.dfg.report.json --cgra-report %t.dir/vecsum.cgra.report.json --dfg-report %t.dir/dotproduct.dfg.report.json --cgra-report %t.dir/dotproduct.cgra.report.json --dfg-report %t.dir/spmv.dfg.report.json --cgra-report %t.dir/spmv.cgra.report.json --dfg-report %t.dir/prefix_sum.dfg.report.json --cgra-report %t.dir/prefix_sum.cgra.report.json --dfg-report %t.dir/cumsum.dfg.report.json --cgra-report %t.dir/cumsum.cgra.report.json --dfg-report %t.dir/integrate_trapz.dfg.report.json --cgra-report %t.dir/integrate_trapz.cgra.report.json --output %t.dir/summary.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-DAG: axpy,136,155,pass
// SUMMARY-DAG: bit_reverse,267,280,pass
// SUMMARY-DAG: conv1d,83,100,pass
// SUMMARY-DAG: convolve_1d,157,180,pass
// SUMMARY-DAG: correlation,346,369,pass
// SUMMARY-DAG: compare_swap,336,366,pass
// SUMMARY-DAG: hash_mix,1280,1305,pass
// SUMMARY-DAG: xor_block,448,466,pass
// SUMMARY-DAG: matvec,371,453,pass
// SUMMARY-DAG: vecadd,1603,1631,pass
// SUMMARY-DAG: vecmul,256,274,pass
// SUMMARY-DAG: vecscale,384,396,pass
// SUMMARY-DAG: mean,904,917,pass
// SUMMARY-DAG: vecnorm_l1,643,654,pass
// SUMMARY-DAG: vecnorm_l2,771,783,pass
// SUMMARY-DAG: reduction,1155,1165,pass
// SUMMARY-DAG: relu,707,731,pass
// SUMMARY-DAG: rotate_bits,544,568,pass
// SUMMARY-DAG: vecsum,579,589,pass
// SUMMARY-DAG: dotproduct,1027,1044,pass
// SUMMARY-DAG: spmv,47,72,pass
// SUMMARY-DAG: prefix_sum,835,852,pass
// SUMMARY-DAG: cumsum,14339,14356,pass
// SUMMARY-DAG: integrate_trapz,299,340,pass
