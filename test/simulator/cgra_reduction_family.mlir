// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/bit_reverse LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/bit_reverse/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/byte_swap LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/byte_swap/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/downsample_avg LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/downsample_avg/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/conv1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/conv1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/convolve_1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/convolve_1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/correlation LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/correlation/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/compare_swap LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/compare_swap/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/hash_mix LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/hash_mix/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/xor_block LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/xor_block/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/axpy LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/axpy/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/relu LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/relu/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/rotate_bits LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/rotate_bits/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/variance LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/variance/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/matvec LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/matvec/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/gemv LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/gemv/dfg_check.sh
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
// RUN: env BUILD_DIR=%t.dir/prefix_sum_inclusive LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/prefix_sum_inclusive/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/cumsum LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/cumsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/integrate_trapz LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/integrate_trapz/dfg_check.sh
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh bit_reverse %t.dir/bit_reverse/main_func.dfg.mlir %t.dir/bit_reverse.dfg.report.json %t.dir/bit_reverse.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh byte_swap %t.dir/byte_swap/main_func.dfg.mlir %t.dir/byte_swap.dfg.report.json %t.dir/byte_swap.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh downsample_avg %t.dir/downsample_avg/main_func.dfg.mlir %t.dir/downsample_avg.dfg.report.json %t.dir/downsample_avg.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh conv1d %t.dir/conv1d/main_func.dfg.mlir %t.dir/conv1d.dfg.report.json %t.dir/conv1d.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh convolve_1d %t.dir/convolve_1d/main_func.dfg.mlir %t.dir/convolve_1d.dfg.report.json %t.dir/convolve_1d.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh correlation %t.dir/correlation/main_func.dfg.mlir %t.dir/correlation.dfg.report.json %t.dir/correlation.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh compare_swap %t.dir/compare_swap/main_func.dfg.mlir %t.dir/compare_swap.dfg.report.json %t.dir/compare_swap.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh hash_mix %t.dir/hash_mix/main_func.dfg.mlir %t.dir/hash_mix.dfg.report.json %t.dir/hash_mix.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh xor_block %t.dir/xor_block/main_func.dfg.mlir %t.dir/xor_block.dfg.report.json %t.dir/xor_block.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh axpy %t.dir/axpy/main_func.dfg.mlir %t.dir/axpy.dfg.report.json %t.dir/axpy.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh relu %t.dir/relu/main_func.dfg.mlir %t.dir/relu.dfg.report.json %t.dir/relu.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh rotate_bits %t.dir/rotate_bits/main_func.dfg.mlir %t.dir/rotate_bits.dfg.report.json %t.dir/rotate_bits.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh variance %t.dir/variance/main_func.dfg.mlir %t.dir/variance.dfg.report.json %t.dir/variance.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh matvec %t.dir/matvec/main_func.dfg.mlir %t.dir/matvec.dfg.report.json %t.dir/matvec.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh gemv %t.dir/gemv/main_func.dfg.mlir %t.dir/gemv.dfg.report.json %t.dir/gemv.dfg.summary.csv
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
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh prefix_sum_inclusive %t.dir/prefix_sum_inclusive/main_func.dfg.mlir %t.dir/prefix_sum_inclusive.dfg.report.json %t.dir/prefix_sum_inclusive.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh cumsum %t.dir/cumsum/main_func.dfg.mlir %t.dir/cumsum.dfg.report.json %t.dir/cumsum.dfg.summary.csv
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh integrate_trapz %t.dir/integrate_trapz/main_func.dfg.mlir %t.dir/integrate_trapz.dfg.report.json %t.dir/integrate_trapz.dfg.summary.csv
// RUN: loom-pnr-map --dfg-mlir %t.dir/bit_reverse/main_func.dfg.mlir --graph g_t_bit_reverse_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload bit_reverse --output %t.dir/bit_reverse.mapping.csv --artifact %t.dir/bit_reverse.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload byte_swap --output %t.dir/byte_swap.mapping.csv --artifact %t.dir/byte_swap.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/downsample_avg/main_func.dfg.mlir --graph g_t_downsample_avg_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload downsample_avg --output %t.dir/downsample_avg.mapping.csv --artifact %t.dir/downsample_avg.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/downsample_avg/main_func.dfg.mlir --graph g_t_main_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload downsample_avg --output %t.dir/downsample_avg.init.mapping.csv --artifact %t.dir/downsample_avg.init.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/conv1d/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload conv1d --output %t.dir/conv1d.mapping.csv --artifact %t.dir/conv1d.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/convolve_1d/main_func.dfg.mlir --graph g_t_convolve_1d_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload convolve_1d --output %t.dir/convolve_1d.mapping.csv --artifact %t.dir/convolve_1d.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/correlation/main_func.dfg.mlir --graph g_t_correlation_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload correlation --output %t.dir/correlation.mapping.csv --artifact %t.dir/correlation.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/compare_swap/main_func.dfg.mlir --graph g_t_main_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload compare_swap --output %t.dir/compare_swap.mapping.csv --artifact %t.dir/compare_swap.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/hash_mix/main_func.dfg.mlir --graph g_t_main_1_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload hash_mix --output %t.dir/hash_mix.mapping.csv --artifact %t.dir/hash_mix.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/xor_block/main_func.dfg.mlir --graph g_t_xor_block_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload xor_block --output %t.dir/xor_block.mapping.csv --artifact %t.dir/xor_block.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/axpy/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_114axpy_candidateEPKjS1_Pjjj_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload axpy --output %t.dir/axpy.mapping.csv --artifact %t.dir/axpy.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/relu/main_func.dfg.mlir --graph g_t_relu_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload relu --output %t.dir/relu.core.mapping.csv --artifact %t.dir/relu.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/relu/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload relu --output %t.dir/relu.checksum.mapping.csv --artifact %t.dir/relu.checksum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/rotate_bits/main_func.dfg.mlir --graph g_t_rotate_bits_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload rotate_bits --output %t.dir/rotate_bits.mapping.csv --artifact %t.dir/rotate_bits.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/variance/main_func.dfg.mlir --graph g_t_variance_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload variance --output %t.dir/variance.mean.mapping.csv --artifact %t.dir/variance.mean.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/variance/main_func.dfg.mlir --graph g_t_variance_red_1_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload variance --output %t.dir/variance.var.mapping.csv --artifact %t.dir/variance.var.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_matvec_kernel_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.core.mapping.csv --artifact %t.dir/matvec.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.checksum.mapping.csv --artifact %t.dir/matvec.checksum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/gemv/main_func.dfg.mlir --graph g_t_gemv_kernel_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload gemv --output %t.dir/gemv.core.mapping.csv --artifact %t.dir/gemv.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/gemv/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload gemv --output %t.dir/gemv.checksum.mapping.csv --artifact %t.dir/gemv.checksum.mapping.json
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
// RUN: loom-pnr-map --dfg-mlir %t.dir/prefix_sum_inclusive/main_func.dfg.mlir --graph g_t_prefix_sum_inclusive_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload prefix_sum_inclusive --output %t.dir/prefix_sum_inclusive.mapping.csv --artifact %t.dir/prefix_sum_inclusive.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/cumsum/main_func.dfg.mlir --graph g_t_cumsum_kernel_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload cumsum --output %t.dir/cumsum.mapping.csv --artifact %t.dir/cumsum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/integrate_trapz/main_func.dfg.mlir --graph g_t_integrate_trapz_red_0_0 --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload integrate_trapz --output %t.dir/integrate_trapz.mapping.csv --artifact %t.dir/integrate_trapz.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/bit_reverse.dfg.report.json --mapping-artifact %t.dir/bit_reverse.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/bit_reverse.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/byte_swap.dfg.report.json --mapping-artifact %t.dir/byte_swap.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/byte_swap.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/downsample_avg.dfg.init.report.json --mapping-artifact %t.dir/downsample_avg.init.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/downsample_avg.init.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/downsample_avg.dfg.report.json --mapping-artifact %t.dir/downsample_avg.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/downsample_avg.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/downsample_avg.dfg.row1.report.json --mapping-artifact %t.dir/downsample_avg.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/downsample_avg.row1.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/downsample_avg.dfg.row2.report.json --mapping-artifact %t.dir/downsample_avg.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/downsample_avg.row2.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/downsample_avg.dfg.row3.report.json --mapping-artifact %t.dir/downsample_avg.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/downsample_avg.row3.cgra.report.json
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
// RUN: loom-cgra-sim --dfg-report %t.dir/variance.dfg.report.json --mapping-artifact %t.dir/variance.mean.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/variance.mean.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/variance.dfg.var.report.json --mapping-artifact %t.dir/variance.var.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/variance.var.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.row1.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.row1.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.row2.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.row2.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.row3.report.json --mapping-artifact %t.dir/matvec.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.row3.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/matvec.dfg.checksum.report.json --mapping-artifact %t.dir/matvec.checksum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/matvec.checksum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/gemv.dfg.report.json --mapping-artifact %t.dir/gemv.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/gemv.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/gemv.dfg.row1.report.json --mapping-artifact %t.dir/gemv.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/gemv.row1.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/gemv.dfg.row2.report.json --mapping-artifact %t.dir/gemv.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/gemv.row2.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/gemv.dfg.row3.report.json --mapping-artifact %t.dir/gemv.core.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/gemv.row3.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/gemv.dfg.checksum.report.json --mapping-artifact %t.dir/gemv.checksum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/gemv.checksum.cgra.report.json
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
// RUN: loom-cgra-sim --dfg-report %t.dir/prefix_sum_inclusive.dfg.report.json --mapping-artifact %t.dir/prefix_sum_inclusive.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/prefix_sum_inclusive.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/cumsum.dfg.report.json --mapping-artifact %t.dir/cumsum.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/cumsum.cgra.report.json
// RUN: loom-cgra-sim --dfg-report %t.dir/integrate_trapz.dfg.report.json --mapping-artifact %t.dir/integrate_trapz.mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/integrate_trapz.cgra.report.json
// RUN: FileCheck %s --check-prefix=MEAN-CGRA < %t.dir/mean.cgra.report.json
// RUN: FileCheck %s --check-prefix=DOWNSAMPLE-CORE-CGRA < %t.dir/downsample_avg.cgra.report.json
// RUN: FileCheck %s --check-prefix=VARIANCE-MEAN-CGRA < %t.dir/variance.mean.cgra.report.json
// RUN: FileCheck %s --check-prefix=MATVEC-CGRA < %t.dir/matvec.cgra.report.json
// RUN: FileCheck %s --check-prefix=VECNORM-L1-CGRA < %t.dir/vecnorm_l1.cgra.report.json
// RUN: FileCheck %s --check-prefix=VECNORM-L2-CGRA < %t.dir/vecnorm_l2.cgra.report.json
// RUN: bash %S/../app/run_sim_cycle_summary.sh --dfg-report %t.dir/axpy.dfg.report.json --cgra-report %t.dir/axpy.cgra.report.json --dfg-report %t.dir/bit_reverse.dfg.report.json --cgra-report %t.dir/bit_reverse.cgra.report.json --dfg-report %t.dir/byte_swap.dfg.report.json --cgra-report %t.dir/byte_swap.cgra.report.json --dfg-report %t.dir/downsample_avg.dfg.init.report.json --dfg-report %t.dir/downsample_avg.dfg.report.json --dfg-report %t.dir/downsample_avg.dfg.row1.report.json --dfg-report %t.dir/downsample_avg.dfg.row2.report.json --dfg-report %t.dir/downsample_avg.dfg.row3.report.json --cgra-report %t.dir/downsample_avg.init.cgra.report.json --cgra-report %t.dir/downsample_avg.cgra.report.json --cgra-report %t.dir/downsample_avg.row1.cgra.report.json --cgra-report %t.dir/downsample_avg.row2.cgra.report.json --cgra-report %t.dir/downsample_avg.row3.cgra.report.json --dfg-report %t.dir/conv1d.dfg.report.json --cgra-report %t.dir/conv1d.cgra.report.json --dfg-report %t.dir/convolve_1d.dfg.report.json --cgra-report %t.dir/convolve_1d.cgra.report.json --dfg-report %t.dir/correlation.dfg.report.json --cgra-report %t.dir/correlation.cgra.report.json --dfg-report %t.dir/compare_swap.dfg.report.json --cgra-report %t.dir/compare_swap.cgra.report.json --dfg-report %t.dir/hash_mix.dfg.report.json --cgra-report %t.dir/hash_mix.cgra.report.json --dfg-report %t.dir/xor_block.dfg.report.json --cgra-report %t.dir/xor_block.cgra.report.json --dfg-report %t.dir/relu.dfg.report.json --dfg-report %t.dir/relu.dfg.checksum.report.json --cgra-report %t.dir/relu.cgra.report.json --cgra-report %t.dir/relu.checksum.cgra.report.json --dfg-report %t.dir/rotate_bits.dfg.report.json --cgra-report %t.dir/rotate_bits.cgra.report.json --dfg-report %t.dir/variance.dfg.report.json --dfg-report %t.dir/variance.dfg.var.report.json --cgra-report %t.dir/variance.mean.cgra.report.json --cgra-report %t.dir/variance.var.cgra.report.json --dfg-report %t.dir/matvec.dfg.report.json --dfg-report %t.dir/matvec.dfg.row1.report.json --dfg-report %t.dir/matvec.dfg.row2.report.json --dfg-report %t.dir/matvec.dfg.row3.report.json --dfg-report %t.dir/matvec.dfg.checksum.report.json --cgra-report %t.dir/matvec.cgra.report.json --cgra-report %t.dir/matvec.row1.cgra.report.json --cgra-report %t.dir/matvec.row2.cgra.report.json --cgra-report %t.dir/matvec.row3.cgra.report.json --cgra-report %t.dir/matvec.checksum.cgra.report.json --dfg-report %t.dir/gemv.dfg.report.json --dfg-report %t.dir/gemv.dfg.row1.report.json --dfg-report %t.dir/gemv.dfg.row2.report.json --dfg-report %t.dir/gemv.dfg.row3.report.json --dfg-report %t.dir/gemv.dfg.checksum.report.json --cgra-report %t.dir/gemv.cgra.report.json --cgra-report %t.dir/gemv.row1.cgra.report.json --cgra-report %t.dir/gemv.row2.cgra.report.json --cgra-report %t.dir/gemv.row3.cgra.report.json --cgra-report %t.dir/gemv.checksum.cgra.report.json --dfg-report %t.dir/vecadd.dfg.report.json --dfg-report %t.dir/vecadd.dfg.reduction.report.json --cgra-report %t.dir/vecadd.core.cgra.report.json --cgra-report %t.dir/vecadd.reduction.cgra.report.json --dfg-report %t.dir/vecmul.dfg.report.json --cgra-report %t.dir/vecmul.cgra.report.json --dfg-report %t.dir/vecscale.dfg.report.json --cgra-report %t.dir/vecscale.cgra.report.json --dfg-report %t.dir/mean.dfg.report.json --cgra-report %t.dir/mean.cgra.report.json --dfg-report %t.dir/vecnorm_l1.dfg.report.json --cgra-report %t.dir/vecnorm_l1.cgra.report.json --dfg-report %t.dir/vecnorm_l2.dfg.report.json --cgra-report %t.dir/vecnorm_l2.cgra.report.json --dfg-report %t.dir/reduction.dfg.report.json --cgra-report %t.dir/reduction.cgra.report.json --dfg-report %t.dir/vecsum.dfg.report.json --cgra-report %t.dir/vecsum.cgra.report.json --dfg-report %t.dir/dotproduct.dfg.report.json --cgra-report %t.dir/dotproduct.cgra.report.json --dfg-report %t.dir/spmv.dfg.report.json --cgra-report %t.dir/spmv.cgra.report.json --dfg-report %t.dir/prefix_sum.dfg.report.json --cgra-report %t.dir/prefix_sum.cgra.report.json --dfg-report %t.dir/prefix_sum_inclusive.dfg.report.json --cgra-report %t.dir/prefix_sum_inclusive.cgra.report.json --dfg-report %t.dir/cumsum.dfg.report.json --cgra-report %t.dir/cumsum.cgra.report.json --dfg-report %t.dir/integrate_trapz.dfg.report.json --cgra-report %t.dir/integrate_trapz.cgra.report.json --output %t.dir/summary.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/summary.csv

// MEAN-CGRA-DAG: "workload": "mean"
// MEAN-CGRA-DAG: "hardware": "shared_reduction_adg"
// MEAN-CGRA-DAG: "status": "pass"
// MEAN-CGRA-DAG: "mapping_id": "mean__g_t_mean_kernel_red_0_0__shared_reduction_adg"
// MEAN-CGRA-DAG: "dfg_cycles": 904
// MEAN-CGRA-DAG: "hardware_aware_cycles": 939
// MEAN-CGRA-DAG: "routed_edges": 9
// MEAN-CGRA-DAG: "route_segments": 31
// MEAN-CGRA-DAG: "fidelity_level": "mapping_constraint_estimate"
// MEAN-CGRA-DAG: "functional_state_source": "carried_from_dfg_sim_report"

// DOWNSAMPLE-CORE-CGRA-DAG: "workload": "downsample_avg"
// DOWNSAMPLE-CORE-CGRA-DAG: "status": "pass"
// DOWNSAMPLE-CORE-CGRA-DAG: "mapping_id": "downsample_avg__g_t_downsample_avg_0_0__shared_reduction_adg"
// DOWNSAMPLE-CORE-CGRA-DAG: "dfg_cycles": 64
// DOWNSAMPLE-CORE-CGRA-DAG: "hardware_aware_cycles": 99
// DOWNSAMPLE-CORE-CGRA-DAG: "routed_edges": 9

// VARIANCE-MEAN-CGRA-DAG: "workload": "variance"
// VARIANCE-MEAN-CGRA-DAG: "status": "pass"
// VARIANCE-MEAN-CGRA-DAG: "mapping_id": "variance__g_t_variance_red_0_0__shared_reduction_adg"
// VARIANCE-MEAN-CGRA-DAG: "dfg_cycles": 232
// VARIANCE-MEAN-CGRA-DAG: "hardware_aware_cycles": 267
// VARIANCE-MEAN-CGRA-DAG: "routed_edges": 9

// MATVEC-CGRA-DAG: "workload": "matvec"
// MATVEC-CGRA-DAG: "hardware": "shared_reduction_adg"
// MATVEC-CGRA-DAG: "status": "pass"
// MATVEC-CGRA-DAG: "mapping_id": "matvec__g_t_matvec_kernel_0_0__shared_reduction_adg"
// MATVEC-CGRA-DAG: "dfg_cycles": 83
// MATVEC-CGRA-DAG: "hardware_aware_cycles": 127
// MATVEC-CGRA-DAG: "routed_edges": 10
// MATVEC-CGRA-DAG: "route_segments": 36
// MATVEC-CGRA-DAG: "fidelity_level": "mapping_constraint_estimate"

// VECNORM-L1-CGRA-DAG: "workload": "vecnorm_l1"
// VECNORM-L1-CGRA-DAG: "hardware": "shared_reduction_adg"
// VECNORM-L1-CGRA-DAG: "status": "pass"
// VECNORM-L1-CGRA-DAG: "mapping_id": "vecnorm_l1__g_t_vecnorm_l1_red_0_0__shared_reduction_adg"
// VECNORM-L1-CGRA-DAG: "dfg_cycles": 643
// VECNORM-L1-CGRA-DAG: "hardware_aware_cycles": {{[0-9]+}}
// VECNORM-L1-CGRA-DAG: "fidelity_level": "mapping_constraint_estimate"

// VECNORM-L2-CGRA-DAG: "workload": "vecnorm_l2"
// VECNORM-L2-CGRA-DAG: "hardware": "shared_reduction_adg"
// VECNORM-L2-CGRA-DAG: "status": "pass"
// VECNORM-L2-CGRA-DAG: "mapping_id": "vecnorm_l2__g_t_vecnorm_l2_red_0_0__shared_reduction_adg"
// VECNORM-L2-CGRA-DAG: "dfg_cycles": 771
// VECNORM-L2-CGRA-DAG: "hardware_aware_cycles": {{[0-9]+}}
// VECNORM-L2-CGRA-DAG: "fidelity_level": "mapping_constraint_estimate"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-DAG: axpy,136,,blocked
// SUMMARY-DAG: bit_reverse,267,{{[0-9]+}},pass
// SUMMARY-DAG: byte_swap,320,,blocked
// SUMMARY-DAG: conv1d,98,137,pass
// SUMMARY-DAG: convolve_1d,178,241,pass
// SUMMARY-DAG: correlation,394,457,pass
// SUMMARY-DAG: downsample_avg,480,,blocked
// SUMMARY-DAG: compare_swap,336,{{[0-9]+}},pass
// SUMMARY-DAG: gemv,423,669,pass
// SUMMARY-DAG: hash_mix,1280,{{[0-9]+}},pass
// SUMMARY-DAG: xor_block,448,,blocked
// SUMMARY-DAG: matvec,371,573,pass
// SUMMARY-DAG: vecadd,1603,1657,pass
// SUMMARY-DAG: vecmul,256,286,pass
// SUMMARY-DAG: vecscale,384,{{[0-9]+}},pass
// SUMMARY-DAG: mean,904,939,pass
// SUMMARY-DAG: vecnorm_l1,643,{{[0-9]+}},pass
// SUMMARY-DAG: vecnorm_l2,771,{{[0-9]+}},pass
// SUMMARY-DAG: reduction,1155,1181,pass
// SUMMARY-DAG: relu,707,759,pass
// SUMMARY-DAG: rotate_bits,544,{{[0-9]+}},pass
// SUMMARY-DAG: variance,594,682,pass
// SUMMARY-DAG: vecsum,579,605,pass
// SUMMARY-DAG: dotproduct,1219,1258,pass
// SUMMARY-DAG: spmv,47,106,pass
// SUMMARY-DAG: prefix_sum,835,878,pass
// SUMMARY-DAG: prefix_sum_inclusive,13302,13345,pass
// SUMMARY-DAG: cumsum,14339,14380,pass
// SUMMARY-DAG: integrate_trapz,323,436,pass
