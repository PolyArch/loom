// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/bit_reverse LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/bit_reverse/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/byte_swap LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/byte_swap/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/downsample LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/downsample/dfg_check.sh
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
// RUN: env BUILD_DIR=%t.dir/sbox_lookup LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/sbox_lookup/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/variance LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/variance/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/matvec LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/matvec/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/gemv LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/gemv/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/gemm LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/gemm/dfg_check.sh
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
// RUN: loom-pnr-map --dfg-mlir %t.dir/bit_reverse/main_func.dfg.mlir --graph g_t_bit_reverse_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload bit_reverse --output %t.dir/bit_reverse.mapping.csv --artifact %t.dir/bit_reverse.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload byte_swap --output %t.dir/byte_swap.mapping.csv --artifact %t.dir/byte_swap.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/downsample/main_func.dfg.mlir --graph g_t_downsample_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload downsample --output %t.dir/downsample.mapping.csv --artifact %t.dir/downsample.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/downsample_avg/main_func.dfg.mlir --graph g_t_downsample_avg_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload downsample_avg --output %t.dir/downsample_avg.mapping.csv --artifact %t.dir/downsample_avg.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/downsample_avg/main_func.dfg.mlir --graph g_t_main_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload downsample_avg --output %t.dir/downsample_avg.init.mapping.csv --artifact %t.dir/downsample_avg.init.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/conv1d/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload conv1d --output %t.dir/conv1d.mapping.csv --artifact %t.dir/conv1d.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/convolve_1d/main_func.dfg.mlir --graph g_t_convolve_1d_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload convolve_1d --output %t.dir/convolve_1d.mapping.csv --artifact %t.dir/convolve_1d.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/correlation/main_func.dfg.mlir --graph g_t_correlation_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload correlation --output %t.dir/correlation.mapping.csv --artifact %t.dir/correlation.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/compare_swap/main_func.dfg.mlir --graph g_t_main_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload compare_swap --output %t.dir/compare_swap.mapping.csv --artifact %t.dir/compare_swap.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/hash_mix/main_func.dfg.mlir --graph g_t_main_1_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload hash_mix --output %t.dir/hash_mix.mapping.csv --artifact %t.dir/hash_mix.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/xor_block/main_func.dfg.mlir --graph g_t_xor_block_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload xor_block --output %t.dir/xor_block.mapping.csv --artifact %t.dir/xor_block.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/axpy/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_114axpy_candidateEPKjS1_Pjjj_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload axpy --output %t.dir/axpy.mapping.csv --artifact %t.dir/axpy.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/relu/main_func.dfg.mlir --graph g_t_relu_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload relu --output %t.dir/relu.core.mapping.csv --artifact %t.dir/relu.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/relu/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload relu --output %t.dir/relu.checksum.mapping.csv --artifact %t.dir/relu.checksum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/rotate_bits/main_func.dfg.mlir --graph g_t_rotate_bits_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload rotate_bits --output %t.dir/rotate_bits.mapping.csv --artifact %t.dir/rotate_bits.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/sbox_lookup/main_func.dfg.mlir --graph g_t_main_2_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload sbox_lookup --output %t.dir/sbox_lookup.mapping.csv --artifact %t.dir/sbox_lookup.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/variance/main_func.dfg.mlir --graph g_t_variance_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload variance --output %t.dir/variance.mean.mapping.csv --artifact %t.dir/variance.mean.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/variance/main_func.dfg.mlir --graph g_t_variance_red_1_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload variance --output %t.dir/variance.var.mapping.csv --artifact %t.dir/variance.var.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_matvec_kernel_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.core.mapping.csv --artifact %t.dir/matvec.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.checksum.mapping.csv --artifact %t.dir/matvec.checksum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/gemv/main_func.dfg.mlir --graph g_t_gemv_kernel_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload gemv --output %t.dir/gemv.core.mapping.csv --artifact %t.dir/gemv.core.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/gemv/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload gemv --output %t.dir/gemv.checksum.mapping.csv --artifact %t.dir/gemv.checksum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/gemm/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_14gemmEPKfS1_Pfiii_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload gemm --output %t.dir/gemm.mapping.csv --artifact %t.dir/gemm.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_main_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/vecadd.mapping.csv --artifact %t.dir/vecadd.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecmul/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecmul --output %t.dir/vecmul.mapping.csv --artifact %t.dir/vecmul.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecscale/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_118vecscale_candidateEPKjjPjj_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecscale --output %t.dir/vecscale.mapping.csv --artifact %t.dir/vecscale.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/mean/main_func.dfg.mlir --graph g_t_mean_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mean --output %t.dir/mean.mapping.csv --artifact %t.dir/mean.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l1/main_func.dfg.mlir --graph g_t_vecnorm_l1_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecnorm_l1 --output %t.dir/vecnorm_l1.mapping.csv --artifact %t.dir/vecnorm_l1.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l2/main_func.dfg.mlir --graph g_t_vecnorm_l2_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecnorm_l2 --output %t.dir/vecnorm_l2.mapping.csv --artifact %t.dir/vecnorm_l2.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/reduction/main_func.dfg.mlir --graph g_t_reduce_sum_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload reduction --output %t.dir/reduction.mapping.csv --artifact %t.dir/reduction.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/vecsum.mapping.csv --artifact %t.dir/vecsum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/dotproduct/main_func.dfg.mlir --graph g_t_dotproduct_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload dotproduct --output %t.dir/dotproduct.mapping.csv --artifact %t.dir/dotproduct.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/spmv/main_func.dfg.mlir --graph g_t_spmv_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload spmv --output %t.dir/spmv.mapping.csv --artifact %t.dir/spmv.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/prefix_sum/main_func.dfg.mlir --graph g_t_prefix_sum_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload prefix_sum --output %t.dir/prefix_sum.mapping.csv --artifact %t.dir/prefix_sum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/prefix_sum_inclusive/main_func.dfg.mlir --graph g_t_prefix_sum_inclusive_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload prefix_sum_inclusive --output %t.dir/prefix_sum_inclusive.mapping.csv --artifact %t.dir/prefix_sum_inclusive.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/cumsum/main_func.dfg.mlir --graph g_t_cumsum_kernel_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload cumsum --output %t.dir/cumsum.mapping.csv --artifact %t.dir/cumsum.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/integrate_trapz/main_func.dfg.mlir --graph g_t_integrate_trapz_red_0_0 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload integrate_trapz --output %t.dir/integrate_trapz.mapping.csv --artifact %t.dir/integrate_trapz.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,BIT-REVERSE < %t.dir/bit_reverse.mapping.csv
// RUN: FileCheck %s --check-prefix=BIT-REVERSE-JSON < %t.dir/bit_reverse.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,BYTE-SWAP < %t.dir/byte_swap.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,DOWNSAMPLE < %t.dir/downsample.mapping.csv
// RUN: FileCheck %s --check-prefix=DOWNSAMPLE-JSON < %t.dir/downsample.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,DOWNSAMPLE-AVG < %t.dir/downsample_avg.mapping.csv
// RUN: FileCheck %s --check-prefix=DOWNSAMPLE-AVG-JSON < %t.dir/downsample_avg.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,DOWNSAMPLE-AVG-INIT < %t.dir/downsample_avg.init.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,CONV1D < %t.dir/conv1d.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,CONVOLVE-1D < %t.dir/convolve_1d.mapping.csv
// RUN: FileCheck %s --check-prefix=CONVOLVE-1D-JSON < %t.dir/convolve_1d.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,CORRELATION < %t.dir/correlation.mapping.csv
// RUN: FileCheck %s --check-prefix=CORRELATION-JSON < %t.dir/correlation.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,COMPARE-SWAP < %t.dir/compare_swap.mapping.csv
// RUN: FileCheck %s --check-prefix=COMPARE-SWAP-JSON < %t.dir/compare_swap.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,HASH-MIX < %t.dir/hash_mix.mapping.csv
// RUN: FileCheck %s --check-prefix=HASH-MIX-JSON < %t.dir/hash_mix.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,XOR-BLOCK < %t.dir/xor_block.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,AXPY < %t.dir/axpy.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,RELU < %t.dir/relu.core.mapping.csv
// RUN: FileCheck %s --check-prefix=RELU-JSON < %t.dir/relu.core.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,RELU-CHECKSUM < %t.dir/relu.checksum.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,ROTATE-BITS < %t.dir/rotate_bits.mapping.csv
// RUN: FileCheck %s --check-prefix=ROTATE-BITS-JSON < %t.dir/rotate_bits.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,SBOX < %t.dir/sbox_lookup.mapping.csv
// RUN: FileCheck %s --check-prefix=SBOX-JSON < %t.dir/sbox_lookup.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,VARIANCE-MEAN < %t.dir/variance.mean.mapping.csv
// RUN: FileCheck %s --check-prefix=VARIANCE-MEAN-JSON < %t.dir/variance.mean.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,VARIANCE-VAR < %t.dir/variance.var.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,MATVEC < %t.dir/matvec.core.mapping.csv
// RUN: FileCheck %s --check-prefix=MATVEC-JSON < %t.dir/matvec.core.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,MATVEC-CHECKSUM < %t.dir/matvec.checksum.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,GEMV < %t.dir/gemv.core.mapping.csv
// RUN: FileCheck %s --check-prefix=GEMV-JSON < %t.dir/gemv.core.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,GEMV-CHECKSUM < %t.dir/gemv.checksum.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,GEMM < %t.dir/gemm.mapping.csv
// RUN: FileCheck %s --check-prefix=GEMM-JSON < %t.dir/gemm.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,VECADD < %t.dir/vecadd.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,VECMUL < %t.dir/vecmul.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,VECSCALE < %t.dir/vecscale.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,MEAN < %t.dir/mean.mapping.csv
// RUN: FileCheck %s --check-prefix=MEAN-JSON < %t.dir/mean.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,VECNORM-L1 < %t.dir/vecnorm_l1.mapping.csv
// RUN: FileCheck %s --check-prefix=VECNORM-L1-JSON < %t.dir/vecnorm_l1.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,VECNORM-L2 < %t.dir/vecnorm_l2.mapping.csv
// RUN: FileCheck %s --check-prefix=VECNORM-L2-JSON < %t.dir/vecnorm_l2.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,REDUCTION < %t.dir/reduction.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,VECSUM < %t.dir/vecsum.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,DOTPRODUCT < %t.dir/dotproduct.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,SPMV < %t.dir/spmv.mapping.csv
// RUN: FileCheck %s --check-prefixes=CSV,PREFIX-SUM < %t.dir/prefix_sum.mapping.csv
// RUN: FileCheck %s --check-prefix=PREFIX-SUM-JSON < %t.dir/prefix_sum.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,PREFIX-SUM-INCLUSIVE < %t.dir/prefix_sum_inclusive.mapping.csv
// RUN: FileCheck %s --check-prefix=PREFIX-SUM-INCLUSIVE-JSON < %t.dir/prefix_sum_inclusive.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,CUMSUM < %t.dir/cumsum.mapping.csv
// RUN: FileCheck %s --check-prefix=CUMSUM-JSON < %t.dir/cumsum.mapping.json
// RUN: FileCheck %s --check-prefixes=CSV,TRAPZ < %t.dir/integrate_trapz.mapping.csv
// RUN: FileCheck %s --check-prefix=TRAPZ-JSON < %t.dir/integrate_trapz.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// AXPY-NEXT: axpy,shared_reduction_adg,axpy__g_t__ZN12_GLOBAL__N_114axpy_candidateEPKjS1_Pjjj_0_0__shared_reduction_adg,6,6,1,0,fail,unrouted software edges lack Fabric ADG connectivity

// RELU-NEXT: relu,shared_reduction_adg,relu__g_t_relu_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources
// RELU-JSON-DAG: "workload": "relu"
// RELU-JSON-DAG: "hardware": "shared_reduction_adg"
// RELU-JSON-DAG: "status": "pass"
// RELU-JSON-DAG: "placed_records": 5
// RELU-JSON-DAG: "routed_edges": 6
// RELU-JSON-DAG: "unrouted_edges": 0
// RELU-JSON-DAG: "edge_ref": "arith.cmpf#0.result0->arith.select#0.operand0"
// RELU-JSON-DAG: "edge_ref": "arith.select#0.result0->dataflow.store#0.operand2"
// RELU-JSON-DAG: "segment_kind": "resource_edge"
// RELU-JSON-DAG: "segment_kind": "module_path"
// RELU-JSON-NOT: ".out"
// RELU-JSON-NOT: ".in"

// RELU-CHECKSUM-NEXT: relu,shared_reduction_adg,relu__g_t_main_red_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// ROTATE-BITS-NEXT: rotate_bits,shared_reduction_adg,rotate_bits__g_t_rotate_bits_0_0__shared_reduction_adg,8,12,0,0,pass,mapped software graph to fabric resources
// ROTATE-BITS-JSON-DAG: "workload": "rotate_bits"
// ROTATE-BITS-JSON-DAG: "hardware": "shared_reduction_adg"
// ROTATE-BITS-JSON-DAG: "status": "pass"
// ROTATE-BITS-JSON-DAG: "placed_records": 8
// ROTATE-BITS-JSON-DAG: "routed_edges": 12
// ROTATE-BITS-JSON-DAG: "unrouted_edges": 0
// ROTATE-BITS-JSON-DAG: "edge_ref": "dataflow.load#1.result0->llvm.intr.fshl#0.operand0"
// ROTATE-BITS-JSON-DAG: "edge_ref": "dataflow.load#1.result0->llvm.intr.fshl#0.operand1"
// ROTATE-BITS-JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.fshl#0.operand2"
// ROTATE-BITS-JSON-DAG: "edge_ref": "arith.andi#0.result0->arith.cmpi#0.operand0"
// ROTATE-BITS-JSON-DAG: "edge_ref": "arith.cmpi#0.result0->arith.select#0.operand0"
// ROTATE-BITS-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.select#0.operand1"
// ROTATE-BITS-JSON-DAG: "edge_ref": "llvm.intr.fshl#0.result0->arith.select#0.operand2"
// ROTATE-BITS-JSON-DAG: "edge_ref": "arith.select#0.result0->dataflow.store#0.operand2"
// ROTATE-BITS-JSON-DAG: "segment_kind": "resource_edge"
// ROTATE-BITS-JSON-DAG: "segment_kind": "module_path"
// ROTATE-BITS-JSON-NOT: ".out"
// ROTATE-BITS-JSON-NOT: ".in"

// SBOX-NEXT: sbox_lookup,shared_reduction_adg,sbox_lookup__g_t_main_2_0__shared_reduction_adg,6,7,0,0,pass,mapped software graph to fabric resources
// SBOX-JSON-DAG: "workload": "sbox_lookup"
// SBOX-JSON-DAG: "hardware": "shared_reduction_adg"
// SBOX-JSON-DAG: "status": "pass"
// SBOX-JSON-DAG: "placed_records": 6
// SBOX-JSON-DAG: "routed_edges": 7
// SBOX-JSON-DAG: "unrouted_edges": 0
// SBOX-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.andi#0.operand0"
// SBOX-JSON-DAG: "edge_ref": "arith.andi#0.result0->llvm.zext#0.operand0"
// SBOX-JSON-DAG: "edge_ref": "llvm.zext#0.result0->dataflow.load#1.operand1"
// SBOX-JSON-DAG: "edge_ref": "dataflow.load#1.result0->dataflow.store#0.operand2"
// SBOX-JSON-DAG: "segment_kind": "resource_edge"
// SBOX-JSON-DAG: "segment_kind": "module_path"
// SBOX-JSON-NOT: ".out"
// SBOX-JSON-NOT: ".in"

// VARIANCE-MEAN-NEXT: variance,shared_reduction_adg,variance__g_t_variance_red_0_0__shared_reduction_adg,7,9,0,0,pass,mapped software graph to fabric resources

// VARIANCE-MEAN-JSON-DAG: "workload": "variance"
// VARIANCE-MEAN-JSON-DAG: "hardware": "shared_reduction_adg"
// VARIANCE-MEAN-JSON-DAG: "status": "pass"
// VARIANCE-MEAN-JSON-DAG: "placed_records": 7
// VARIANCE-MEAN-JSON-DAG: "routed_edges": 9
// VARIANCE-MEAN-JSON-DAG: "unrouted_edges": 0
// VARIANCE-MEAN-JSON-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.invariant#0.operand0"
// VARIANCE-MEAN-JSON-NOT: ".out"
// VARIANCE-MEAN-JSON-NOT: ".in"

// VARIANCE-VAR-NEXT: variance,shared_reduction_adg,variance__g_t_variance_red_1_0__shared_reduction_adg,9,13,0,0,pass,mapped software graph to fabric resources

// BIT-REVERSE-NEXT: bit_reverse,shared_reduction_adg,bit_reverse__g_t_bit_reverse_kernel_red_0_0__shared_reduction_adg,8,13,0,0,pass,mapped software graph to fabric resources
// BIT-REVERSE-JSON-DAG: "workload": "bit_reverse"
// BIT-REVERSE-JSON-DAG: "hardware": "shared_reduction_adg"
// BIT-REVERSE-JSON-DAG: "status": "pass"
// BIT-REVERSE-JSON-DAG: "placed_records": 8
// BIT-REVERSE-JSON-DAG: "routed_edges": 13
// BIT-REVERSE-JSON-DAG: "unrouted_edges": 0
// BIT-REVERSE-JSON-DAG: "edge_ref": "arith.ori#0.result0->dataflow.carry#0.operand2"
// BIT-REVERSE-JSON-DAG: "edge_ref": "arith.shli#0.result0->arith.ori#0.operand0"
// BIT-REVERSE-JSON-DAG: "edge_ref": "arith.shrui#0.result0->dataflow.carry#1.operand2"
// BIT-REVERSE-JSON-DAG: "edge_ref": "dataflow.carry#1.result0->arith.andi#0.operand0"
// BIT-REVERSE-JSON-DAG: "edge_ref": "dataflow.carry#1.result0->arith.shrui#0.operand0"
// BIT-REVERSE-JSON-DAG: "edge_ref": "dataflow.invariant#0.result0->arith.shrui#0.operand1"
// BIT-REVERSE-JSON-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.carry#1.operand0"
// BIT-REVERSE-JSON-DAG: "segment_kind": "resource_edge"
// BIT-REVERSE-JSON-DAG: "segment_kind": "module_path"
// BIT-REVERSE-JSON-NOT: ".out"
// BIT-REVERSE-JSON-NOT: ".in"

// BYTE-SWAP-NEXT: byte_swap,shared_reduction_adg,byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__shared_reduction_adg,4,2,2,0,fail,unrouted software edges lack Fabric ADG connectivity

// DOWNSAMPLE-NEXT: downsample,shared_reduction_adg,downsample__g_t_downsample_0_0__shared_reduction_adg,6,6,0,0,pass,mapped software graph to fabric resources

// DOWNSAMPLE-JSON-DAG: "workload": "downsample"
// DOWNSAMPLE-JSON-DAG: "hardware": "shared_reduction_adg"
// DOWNSAMPLE-JSON-DAG: "status": "pass"
// DOWNSAMPLE-JSON-DAG: "placed_records": 6
// DOWNSAMPLE-JSON-DAG: "routed_edges": 6
// DOWNSAMPLE-JSON-DAG: "unrouted_edges": 0
// DOWNSAMPLE-JSON-DAG: "edge_ref": "dataflow.constant#0.result0->arith.shrui#0.operand1"
// DOWNSAMPLE-JSON-DAG: "edge_ref": "arith.shli#0.result0->arith.shrui#0.operand0"
// DOWNSAMPLE-JSON-DAG: "edge_ref": "arith.shrui#0.result0->dataflow.load#0.operand1"
// DOWNSAMPLE-JSON-DAG: "edge_ref": "dataflow.load#0.result0->dataflow.store#0.operand2"
// DOWNSAMPLE-JSON-DAG: "segment_kind": "module_path"
// DOWNSAMPLE-JSON-NOT: ".out"
// DOWNSAMPLE-JSON-NOT: ".in"

// DOWNSAMPLE-AVG-NEXT: downsample_avg,shared_reduction_adg,downsample_avg__g_t_downsample_avg_0_0__shared_reduction_adg,7,9,0,0,pass,mapped software graph to fabric resources

// DOWNSAMPLE-AVG-JSON-DAG: "workload": "downsample_avg"
// DOWNSAMPLE-AVG-JSON-DAG: "hardware": "shared_reduction_adg"
// DOWNSAMPLE-AVG-JSON-DAG: "status": "pass"
// DOWNSAMPLE-AVG-JSON-DAG: "placed_records": 7
// DOWNSAMPLE-AVG-JSON-DAG: "routed_edges": 9
// DOWNSAMPLE-AVG-JSON-DAG: "unrouted_edges": 0
// DOWNSAMPLE-AVG-JSON-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.invariant#0.operand0"
// DOWNSAMPLE-AVG-JSON-NOT: ".out"
// DOWNSAMPLE-AVG-JSON-NOT: ".in"

// DOWNSAMPLE-AVG-INIT-NEXT: downsample_avg,shared_reduction_adg,downsample_avg__g_t_main_0_0__shared_reduction_adg,5,1,3,1,fail,missing hardware resource for software op llvm.trunc

// CONV1D-NEXT: conv1d,shared_reduction_adg,conv1d__g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0__shared_reduction_adg,6,9,0,0,pass,mapped software graph to fabric resources

// CONVOLVE-1D-NEXT: convolve_1d,shared_reduction_adg,convolve_1d__g_t_convolve_1d_kernel_red_0_0__shared_reduction_adg,10,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources
// CONVOLVE-1D-JSON-DAG: "workload": "convolve_1d"
// CONVOLVE-1D-JSON-DAG: "hardware": "shared_reduction_adg"
// CONVOLVE-1D-JSON-DAG: "status": "pass"
// CONVOLVE-1D-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->arith.addi#0.operand0"
// CONVOLVE-1D-JSON-DAG: "edge_ref": "dataflow.invariant#1.result0->arith.addi#0.operand1"
// CONVOLVE-1D-JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.andi#0.operand0"
// CONVOLVE-1D-JSON-DAG: "edge_ref": "dataflow.invariant#0.result0->arith.andi#0.operand1"
// CONVOLVE-1D-JSON-DAG: "edge_ref": "arith.andi#0.result0->dataflow.load#0.operand1"
// CONVOLVE-1D-JSON-NOT: ".out"
// CONVOLVE-1D-JSON-NOT: ".in"

// CORRELATION-NEXT: correlation,shared_reduction_adg,correlation__g_t_correlation_kernel_red_0_0__shared_reduction_adg,10,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources
// CORRELATION-JSON-DAG: "workload": "correlation"
// CORRELATION-JSON-DAG: "hardware": "shared_reduction_adg"
// CORRELATION-JSON-DAG: "status": "pass"
// CORRELATION-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->arith.addi#0.operand0"
// CORRELATION-JSON-DAG: "edge_ref": "dataflow.invariant#1.result0->arith.addi#0.operand1"
// CORRELATION-JSON-DAG: "edge_ref": "arith.addi#0.result0->arith.andi#0.operand0"
// CORRELATION-JSON-DAG: "edge_ref": "dataflow.invariant#0.result0->arith.andi#0.operand1"
// CORRELATION-JSON-DAG: "edge_ref": "arith.andi#0.result0->dataflow.load#0.operand1"
// CORRELATION-JSON-NOT: ".out"
// CORRELATION-JSON-NOT: ".in"

// COMPARE-SWAP-NEXT: compare_swap,shared_reduction_adg,compare_swap__g_t_main_0_0__shared_reduction_adg,8,14,0,0,pass,mapped software graph to fabric resources
// COMPARE-SWAP-JSON-DAG: "workload": "compare_swap"
// COMPARE-SWAP-JSON-DAG: "hardware": "shared_reduction_adg"
// COMPARE-SWAP-JSON-DAG: "status": "pass"
// COMPARE-SWAP-JSON-DAG: "placed_records": 8
// COMPARE-SWAP-JSON-DAG: "routed_edges": 14
// COMPARE-SWAP-JSON-DAG: "unrouted_edges": 0
// COMPARE-SWAP-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.cmpf#0.operand1"
// COMPARE-SWAP-JSON-DAG: "edge_ref": "arith.select#1.result0->dataflow.store#1.operand2"
// COMPARE-SWAP-JSON-DAG: "edge_ref": "dataflow.store#1.result0->dataflow.sync#0.operand3"
// COMPARE-SWAP-JSON-DAG: "segment_kind": "resource_edge"
// COMPARE-SWAP-JSON-DAG: "segment_kind": "module_path"
// COMPARE-SWAP-JSON-NOT: ".out"
// COMPARE-SWAP-JSON-NOT: ".in"

// HASH-MIX-NEXT: hash_mix,shared_reduction_adg,hash_mix__g_t_main_1_0__shared_reduction_adg,9,13,0,0,pass,mapped software graph to fabric resources
// HASH-MIX-JSON-DAG: "workload": "hash_mix"
// HASH-MIX-JSON-DAG: "hardware": "shared_reduction_adg"
// HASH-MIX-JSON-DAG: "status": "pass"
// HASH-MIX-JSON-DAG: "placed_records": 9
// HASH-MIX-JSON-DAG: "routed_edges": 13
// HASH-MIX-JSON-DAG: "unrouted_edges": 0
// HASH-MIX-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addi#0.operand1"
// HASH-MIX-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.addi#0.operand0"
// HASH-MIX-JSON-DAG: "edge_ref": "arith.addi#0.result0->llvm.intr.fshl#0.operand0"
// HASH-MIX-JSON-DAG: "edge_ref": "arith.addi#0.result0->llvm.intr.fshl#0.operand1"
// HASH-MIX-JSON-DAG: "edge_ref": "llvm.intr.fshl#0.result0->arith.xori#0.operand0"
// HASH-MIX-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.xori#0.operand1"
// HASH-MIX-JSON-DAG: "edge_ref": "arith.xori#0.result0->arith.muli#0.operand0"
// HASH-MIX-JSON-DAG: "edge_ref": "arith.muli#0.result0->llvm.intr.fshl#1.operand0"
// HASH-MIX-JSON-DAG: "edge_ref": "arith.muli#0.result0->llvm.intr.fshl#1.operand1"
// HASH-MIX-JSON-DAG: "edge_ref": "llvm.intr.fshl#1.result0->dataflow.store#0.operand2"
// HASH-MIX-JSON-DAG: "segment_kind": "resource_edge"
// HASH-MIX-JSON-DAG: "segment_kind": "module_path"
// HASH-MIX-JSON-NOT: ".out"
// HASH-MIX-JSON-NOT: ".in"

// XOR-BLOCK-NEXT: xor_block,shared_reduction_adg,xor_block__g_t_xor_block_0_0__shared_reduction_adg,5,4,2,0,fail,unrouted software edges lack Fabric ADG connectivity

// MATVEC-NEXT: matvec,shared_reduction_adg,matvec__g_t_matvec_kernel_0_0__shared_reduction_adg,7,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// MATVEC-JSON-DAG: "workload": "matvec"
// MATVEC-JSON-DAG: "hardware": "shared_reduction_adg"
// MATVEC-JSON-DAG: "status": "pass"
// MATVEC-JSON-DAG: "placed_records": 7
// MATVEC-JSON-DAG: "unrouted_edges": 0
// MATVEC-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.muli#0.operand0"
// MATVEC-JSON-DAG: "edge_ref": "dataflow.load#1.result1->dataflow.sync#0.operand1"
// MATVEC-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#1.operand1"
// MATVEC-JSON-DAG: "segment_kind": "resource_edge"
// MATVEC-JSON-DAG: "segment_kind": "module_path"
// MATVEC-JSON-NOT: ".out"
// MATVEC-JSON-NOT: ".in"

// MATVEC-CHECKSUM-NEXT: matvec,shared_reduction_adg,matvec__g_t_main_red_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// GEMV-NEXT: gemv,shared_reduction_adg,gemv__g_t_gemv_kernel_0_0__shared_reduction_adg,9,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// GEMV-JSON-DAG: "workload": "gemv"
// GEMV-JSON-DAG: "hardware": "shared_reduction_adg"
// GEMV-JSON-DAG: "status": "pass"
// GEMV-JSON-DAG: "placed_records": 9
// GEMV-JSON-DAG: "unrouted_edges": 0
// GEMV-JSON-DAG: "edge_ref": "dataflow.carry#0.result0->arith.shli#0.operand0"
// GEMV-JSON-DAG: "edge_ref": "dataflow.invariant#0.result0->arith.shli#0.operand1"
// GEMV-JSON-DAG: "segment_kind": "resource_edge"
// GEMV-JSON-DAG: "segment_kind": "module_path"
// GEMV-JSON-NOT: ".out"
// GEMV-JSON-NOT: ".in"

// GEMV-CHECKSUM-NEXT: gemv,shared_reduction_adg,gemv__g_t_main_red_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// GEMM-NEXT: gemm,shared_reduction_adg,gemm__g_t__ZN12_GLOBAL__N_14gemmEPKfS1_Pfiii_0_0__shared_reduction_adg,10,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources
// GEMM-JSON-DAG: "workload": "gemm"
// GEMM-JSON-DAG: "hardware": "shared_reduction_adg"
// GEMM-JSON-DAG: "status": "pass"
// GEMM-JSON-DAG: "placed_records": 10
// GEMM-JSON-DAG: "unrouted_edges": 0
// GEMM-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->arith.shli#0.operand0"
// GEMM-JSON-DAG: "edge_ref": "arith.shrui#0.result0->dataflow.load#1.operand1"
// GEMM-JSON-DAG: "segment_kind": "resource_edge"
// GEMM-JSON-DAG: "segment_kind": "module_path"
// GEMM-JSON-NOT: ".out"
// GEMM-JSON-NOT: ".in"

// VECADD-NEXT: vecadd,shared_reduction_adg,vecadd__g_t_main_red_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// VECMUL-NEXT: vecmul,shared_reduction_adg,vecmul__g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0__shared_reduction_adg,5,3,3,0,fail,unrouted software edges lack Fabric ADG connectivity

// VECSCALE-NEXT: vecscale,shared_reduction_adg,vecscale__g_t__ZN12_GLOBAL__N_118vecscale_candidateEPKjjPjj_0_0__shared_reduction_adg,4,3,1,0,fail,unrouted software edges lack Fabric ADG connectivity

// MEAN-NEXT: mean,shared_reduction_adg,mean__g_t_mean_kernel_red_0_0__shared_reduction_adg,7,9,0,0,pass,mapped software graph to fabric resources

// MEAN-JSON-DAG: "workload": "mean"
// MEAN-JSON-DAG: "hardware": "shared_reduction_adg"
// MEAN-JSON-DAG: "status": "pass"
// MEAN-JSON-DAG: "placed_records": 7
// MEAN-JSON-DAG: "routed_edges": 9
// MEAN-JSON-DAG: "unrouted_edges": 0
// MEAN-JSON-DAG: "edge_ref": "dataflow.carry#0.result0->arith.mulf#0.operand0"
// MEAN-JSON-DAG: "edge_ref": "dataflow.invariant#0.result0->arith.mulf#0.operand1"
// MEAN-JSON-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.invariant#0.operand0"
// MEAN-JSON-DAG: "segment_kind": "resource_edge"
// MEAN-JSON-DAG: "segment_kind": "module_path"
// MEAN-JSON-NOT: ".out"
// MEAN-JSON-NOT: ".in"

// VECNORM-L1-NEXT: vecnorm_l1,shared_reduction_adg,vecnorm_l1__g_t_vecnorm_l1_red_0_0__shared_reduction_adg,6,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// VECNORM-L1-JSON-DAG: "workload": "vecnorm_l1"
// VECNORM-L1-JSON-DAG: "hardware": "shared_reduction_adg"
// VECNORM-L1-JSON-DAG: "status": "pass"
// VECNORM-L1-JSON-DAG: "placed_records": 6
// VECNORM-L1-JSON-DAG: "unrouted_edges": 0
// VECNORM-L1-JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.abs#0.operand0"
// VECNORM-L1-JSON-DAG: "edge_ref": "llvm.intr.abs#0.result0->arith.addi#0.operand0"
// VECNORM-L1-JSON-DAG: "segment_kind": "resource_edge"
// VECNORM-L1-JSON-DAG: "segment_kind": "module_path"
// VECNORM-L1-JSON-NOT: ".out"
// VECNORM-L1-JSON-NOT: ".in"

// VECNORM-L2-NEXT: vecnorm_l2,shared_reduction_adg,vecnorm_l2__g_t_vecnorm_l2_red_0_0__shared_reduction_adg,6,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// VECNORM-L2-JSON-DAG: "workload": "vecnorm_l2"
// VECNORM-L2-JSON-DAG: "hardware": "shared_reduction_adg"
// VECNORM-L2-JSON-DAG: "status": "pass"
// VECNORM-L2-JSON-DAG: "placed_records": 6
// VECNORM-L2-JSON-DAG: "unrouted_edges": 0
// VECNORM-L2-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.muli#0.operand0"
// VECNORM-L2-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.muli#0.operand1"
// VECNORM-L2-JSON-DAG: "edge_ref": "arith.muli#0.result0->arith.addi#0.operand0"
// VECNORM-L2-JSON-DAG: "segment_kind": "resource_edge"
// VECNORM-L2-JSON-DAG: "segment_kind": "module_path"
// VECNORM-L2-JSON-NOT: ".out"
// VECNORM-L2-JSON-NOT: ".in"

// REDUCTION-NEXT: reduction,shared_reduction_adg,reduction__g_t_reduce_sum_red_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// VECSUM-NEXT: vecsum,shared_reduction_adg,vecsum__g_t_vecsum_red_0_0__shared_reduction_adg,5,6,0,0,pass,mapped software graph to fabric resources

// DOTPRODUCT-NEXT: dotproduct,shared_reduction_adg,dotproduct__g_t_dotproduct_red_0_0__shared_reduction_adg,6,9,0,0,pass,mapped software graph to fabric resources

// SPMV-NEXT: spmv,shared_reduction_adg,spmv__g_t_spmv_kernel_red_0_0__shared_reduction_adg,9,13,0,0,pass,mapped software graph to fabric resources

// PREFIX-SUM-NEXT: prefix_sum,shared_reduction_adg,prefix_sum__g_t_prefix_sum_red_0_0__shared_reduction_adg,6,9,0,0,pass,mapped software graph to fabric resources

// PREFIX-SUM-JSON-DAG: "workload": "prefix_sum"
// PREFIX-SUM-JSON-DAG: "hardware": "shared_reduction_adg"
// PREFIX-SUM-JSON-DAG: "status": "pass"
// PREFIX-SUM-JSON-DAG: "placed_records": 6
// PREFIX-SUM-JSON-DAG: "routed_edges": 9
// PREFIX-SUM-JSON-DAG: "unrouted_edges": 0
// PREFIX-SUM-JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.store#0.operand2"
// PREFIX-SUM-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// PREFIX-SUM-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// PREFIX-SUM-JSON-DAG: "source_endpoint": "shared_reduction_adg::mem.store#0.result0"
// PREFIX-SUM-JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand1"
// PREFIX-SUM-JSON-DAG: "segment_kind": "resource_edge"
// PREFIX-SUM-JSON-DAG: "segment_kind": "module_path"
// PREFIX-SUM-JSON-NOT: ".out"
// PREFIX-SUM-JSON-NOT: ".in"

// PREFIX-SUM-INCLUSIVE-NEXT: prefix_sum_inclusive,shared_reduction_adg,prefix_sum_inclusive__g_t_prefix_sum_inclusive_kernel_red_0_0__shared_reduction_adg,6,9,0,0,pass,mapped software graph to fabric resources

// PREFIX-SUM-INCLUSIVE-JSON-DAG: "workload": "prefix_sum_inclusive"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "hardware": "shared_reduction_adg"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "status": "pass"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "placed_records": 6
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "routed_edges": 9
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "unrouted_edges": 0
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.store#0.operand2"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "source_endpoint": "shared_reduction_adg::mem.store#0.result0"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand1"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "segment_kind": "resource_edge"
// PREFIX-SUM-INCLUSIVE-JSON-DAG: "segment_kind": "module_path"
// PREFIX-SUM-INCLUSIVE-JSON-NOT: ".out"
// PREFIX-SUM-INCLUSIVE-JSON-NOT: ".in"

// CUMSUM-NEXT: cumsum,shared_reduction_adg,cumsum__g_t_cumsum_kernel_red_0_0__shared_reduction_adg,6,{{[1-9][0-9]*}},0,0,pass,mapped software graph to fabric resources

// CUMSUM-JSON-DAG: "workload": "cumsum"
// CUMSUM-JSON-DAG: "hardware": "shared_reduction_adg"
// CUMSUM-JSON-DAG: "status": "pass"
// CUMSUM-JSON-DAG: "placed_records": 6
// CUMSUM-JSON-DAG: "unrouted_edges": 0
// CUMSUM-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.carry#0.operand2"
// CUMSUM-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"
// CUMSUM-JSON-DAG: "edge_ref": "dataflow.carry#0.result0->arith.addf#0.operand0"
// CUMSUM-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand1"
// CUMSUM-JSON-DAG: "segment_kind": "resource_edge"
// CUMSUM-JSON-DAG: "segment_kind": "module_path"
// CUMSUM-JSON-NOT: ".out"
// CUMSUM-JSON-NOT: ".in"

// TRAPZ-NEXT: integrate_trapz,shared_reduction_adg,integrate_trapz__g_t_integrate_trapz_red_0_0__shared_reduction_adg,15,13,12,0,fail,unrouted software edges lack Fabric ADG connectivity
// TRAPZ-JSON-DAG: "workload": "integrate_trapz"
// TRAPZ-JSON-DAG: "hardware": "shared_reduction_adg"
// TRAPZ-JSON-DAG: "status": "fail"
// TRAPZ-JSON-DAG: "unrouted_edges": 12
// TRAPZ-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.subf#0.operand0"
// TRAPZ-JSON-DAG: "source_endpoint": "shared_reduction_adg::mem.load#0.result0"
// TRAPZ-JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.operand1"
// TRAPZ-JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.result0"
// TRAPZ-JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand0"
// TRAPZ-JSON-DAG: "edge_ref": "arith.subf#0.result0->llvm.intr.fmuladd#0.operand1"
// TRAPZ-JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#{{[0-9]+}}.result0"
// TRAPZ-JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.operand2"
// TRAPZ-JSON-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.result0"
// TRAPZ-JSON-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand1"
// TRAPZ-JSON-DAG: "segment_kind": "module_path"
// TRAPZ-JSON-NOT: ".out"
// TRAPZ-JSON-NOT: ".in"
