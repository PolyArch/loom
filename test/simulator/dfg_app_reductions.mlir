// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/bit_reverse LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/bit_reverse/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/conv1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/conv1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/convolve_1d LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/convolve_1d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/correlation LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/correlation/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/cumsum LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/cumsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/compare_swap LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/compare_swap/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/hash_mix LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/hash_mix/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/xor_block LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/xor_block/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/axpy LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/axpy/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/relu LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/relu/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/rotate_bits LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/rotate_bits/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/variance LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/variance/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/matvec LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/matvec/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecmul LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecmul/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecscale LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecscale/dfg_check.sh
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
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh compare_swap %t.dir/compare_swap/main_func.dfg.mlir %t.dir/reports/compare_swap.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh hash_mix %t.dir/hash_mix/main_func.dfg.mlir %t.dir/reports/hash_mix.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh xor_block %t.dir/xor_block/main_func.dfg.mlir %t.dir/reports/xor_block.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh axpy %t.dir/axpy/main_func.dfg.mlir %t.dir/reports/axpy.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh relu %t.dir/relu/main_func.dfg.mlir %t.dir/reports/relu.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh rotate_bits %t.dir/rotate_bits/main_func.dfg.mlir %t.dir/reports/rotate_bits.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh variance %t.dir/variance/main_func.dfg.mlir %t.dir/reports/variance.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh matvec %t.dir/matvec/main_func.dfg.mlir %t.dir/reports/matvec.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecadd %t.dir/vecadd/main_func.dfg.mlir %t.dir/reports/vecadd.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecmul %t.dir/vecmul/main_func.dfg.mlir %t.dir/reports/vecmul.report.json %t.dir/summary.csv --append
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh vecscale %t.dir/vecscale/main_func.dfg.mlir %t.dir/reports/vecscale.report.json %t.dir/summary.csv --append
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
// RUN: FileCheck %s --check-prefix=COMPARE-SWAP < %t.dir/reports/compare_swap.report.json
// RUN: FileCheck %s --check-prefix=HASH-MIX < %t.dir/reports/hash_mix.report.json
// RUN: FileCheck %s --check-prefix=XOR-BLOCK < %t.dir/reports/xor_block.report.json
// RUN: FileCheck %s --check-prefix=AXPY < %t.dir/reports/axpy.report.json
// RUN: FileCheck %s --check-prefix=RELU < %t.dir/reports/relu.report.json
// RUN: FileCheck %s --check-prefix=RELU-CHECKSUM < %t.dir/reports/relu.checksum.report.json
// RUN: FileCheck %s --check-prefix=ROTATE-BITS < %t.dir/reports/rotate_bits.report.json
// RUN: FileCheck %s --check-prefix=VARIANCE-MEAN < %t.dir/reports/variance.report.json
// RUN: FileCheck %s --check-prefix=VARIANCE-VAR < %t.dir/reports/variance.var.report.json
// RUN: FileCheck %s --check-prefix=MATVEC < %t.dir/reports/matvec.report.json
// RUN: FileCheck %s --check-prefix=MATVEC-ROW1 < %t.dir/reports/matvec.row1.report.json
// RUN: FileCheck %s --check-prefix=MATVEC-CHECKSUM < %t.dir/reports/matvec.checksum.report.json
// RUN: FileCheck %s --check-prefix=VECADD < %t.dir/reports/vecadd.report.json
// RUN: FileCheck %s --check-prefix=VECADD-REDUCTION < %t.dir/reports/vecadd.reduction.report.json
// RUN: FileCheck %s --check-prefix=VECMUL < %t.dir/reports/vecmul.report.json
// RUN: FileCheck %s --check-prefix=VECSCALE < %t.dir/reports/vecscale.report.json
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
// CONV1D-DAG: "optimistic_cycles": 83
// CONV1D-DAG: "wavefront_steps": 13
// CONV1D-DAG: "event_count": 38
// CONV1D-DAG: "f32:5"

// CONVOLVE-1D-DAG: "kind": "dfg_sim_report"
// CONVOLVE-1D-DAG: "workload": "convolve_1d"
// CONVOLVE-1D-DAG: "graph": "g_t_convolve_1d_kernel_0_0"
// CONVOLVE-1D-DAG: "status": "pass"
// CONVOLVE-1D-DAG: "optimistic_cycles": 157
// CONVOLVE-1D-DAG: "wavefront_steps": 19
// CONVOLVE-1D-DAG: "event_count": 94
// CONVOLVE-1D-DAG: "f32:1.000000"

// CORRELATION-DAG: "kind": "dfg_sim_report"
// CORRELATION-DAG: "workload": "correlation"
// CORRELATION-DAG: "graph": "g_t_correlation_kernel_0_0"
// CORRELATION-DAG: "status": "pass"
// CORRELATION-DAG: "optimistic_cycles": 346
// CORRELATION-DAG: "wavefront_steps": 37
// CORRELATION-DAG: "event_count": 202
// CORRELATION-DAG: "f32:16"

// CUMSUM-DAG: "kind": "dfg_sim_report"
// CUMSUM-DAG: "workload": "cumsum"
// CUMSUM-DAG: "graph": "g_t_cumsum_kernel_red_0_0"
// CUMSUM-DAG: "status": "pass"
// CUMSUM-DAG: "optimistic_cycles": 14339
// CUMSUM-DAG: "wavefront_steps": 2052
// CUMSUM-DAG: "event_count": 7171
// CUMSUM-DAG: "f32:5620"

// COMPARE-SWAP-DAG: "kind": "dfg_sim_report"
// COMPARE-SWAP-DAG: "workload": "compare_swap"
// COMPARE-SWAP-DAG: "graph": "g_t_main_0_0"
// COMPARE-SWAP-DAG: "status": "pass"
// COMPARE-SWAP-DAG: "optimistic_cycles": 336
// COMPARE-SWAP-DAG: "wavefront_steps": 20
// COMPARE-SWAP-DAG: "event_count": 128
// COMPARE-SWAP-DAG: "dynamic_work_items": 16

// HASH-MIX-DAG: "kind": "dfg_sim_report"
// HASH-MIX-DAG: "workload": "hash_mix"
// HASH-MIX-DAG: "graph": "g_t_main_1_0"
// HASH-MIX-DAG: "status": "pass"
// HASH-MIX-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// HASH-MIX-DAG: "optimistic_cycles": 1280
// HASH-MIX-DAG: "wavefront_steps": 71
// HASH-MIX-DAG: "event_count": 576
// HASH-MIX-DAG: "dynamic_work_items": 64
// HASH-MIX-DAG: "llvm.intr.fshl": 128
// HASH-MIX-DAG: "arith.xori": 64

// XOR-BLOCK-DAG: "kind": "dfg_sim_report"
// XOR-BLOCK-DAG: "workload": "xor_block"
// XOR-BLOCK-DAG: "graph": "g_t_xor_block_0_0"
// XOR-BLOCK-DAG: "status": "pass"
// XOR-BLOCK-DAG: "optimistic_cycles": 448
// XOR-BLOCK-DAG: "wavefront_steps": 35
// XOR-BLOCK-DAG: "event_count": 160
// XOR-BLOCK-DAG: "dynamic_work_items": 32
// XOR-BLOCK-DAG: "arith.xori": 32
// XOR-BLOCK-DAG: "dataflow.load": 64
// XOR-BLOCK-DAG: "dataflow.store": 32
// XOR-BLOCK-DAG: "dataflow.sync": 32

// AXPY-DAG: "kind": "dfg_sim_report"
// AXPY-DAG: "workload": "axpy"
// AXPY-DAG: "graph": "g_t__ZN12_GLOBAL__N_114axpy_candidateEPKjS1_Pjjj_0_0"
// AXPY-DAG: "status": "pass"
// AXPY-DAG: "optimistic_cycles": 136
// AXPY-DAG: "wavefront_steps": 12
// AXPY-DAG: "event_count": 48
// AXPY-DAG: "dynamic_work_items": 8
// AXPY-DAG: "arith.addi": 8
// AXPY-DAG: "arith.muli": 8
// AXPY-DAG: "dataflow.load": 16
// AXPY-DAG: "dataflow.store": 8

// RELU-DAG: "kind": "dfg_sim_report"
// RELU-DAG: "workload": "relu"
// RELU-DAG: "graph": "g_t_relu_0_0"
// RELU-DAG: "status": "pass"
// RELU-DAG: "optimistic_cycles": 384
// RELU-DAG: "wavefront_steps": 36
// RELU-DAG: "event_count": 160
// RELU-DAG: "dynamic_work_items": 32
// RELU-DAG: "arith.cmpf": 32
// RELU-DAG: "arith.select": 32
// RELU-DAG: "dataflow.load": 32
// RELU-DAG: "dataflow.store": 32

// RELU-CHECKSUM-DAG: "kind": "dfg_sim_report"
// RELU-CHECKSUM-DAG: "workload": "relu"
// RELU-CHECKSUM-DAG: "graph": "g_t_main_red_0_0"
// RELU-CHECKSUM-DAG: "status": "pass"
// RELU-CHECKSUM-DAG: "optimistic_cycles": 323
// RELU-CHECKSUM-DAG: "dynamic_work_items": 32
// RELU-CHECKSUM-DAG: "f32:42"

// ROTATE-BITS-DAG: "kind": "dfg_sim_report"
// ROTATE-BITS-DAG: "workload": "rotate_bits"
// ROTATE-BITS-DAG: "graph": "g_t_rotate_bits_0_0"
// ROTATE-BITS-DAG: "status": "pass"
// ROTATE-BITS-DAG: "optimistic_cycles": 544
// ROTATE-BITS-DAG: "wavefront_steps": 37
// ROTATE-BITS-DAG: "event_count": 256
// ROTATE-BITS-DAG: "dynamic_work_items": 32
// ROTATE-BITS-DAG: "arith.andi": 32
// ROTATE-BITS-DAG: "arith.cmpi": 32
// ROTATE-BITS-DAG: "llvm.intr.fshl": 32
// ROTATE-BITS-DAG: "arith.select": 32
// ROTATE-BITS-DAG: "dataflow.load": 64
// ROTATE-BITS-DAG: "dataflow.store": 32

// VARIANCE-MEAN-DAG: "kind": "dfg_sim_report"
// VARIANCE-MEAN-DAG: "workload": "variance"
// VARIANCE-MEAN-DAG: "graph": "g_t_variance_red_0_0"
// VARIANCE-MEAN-DAG: "status": "pass"
// VARIANCE-MEAN-DAG: "optimistic_cycles": 232
// VARIANCE-MEAN-DAG: "wavefront_steps": 36
// VARIANCE-MEAN-DAG: "event_count": 134
// VARIANCE-MEAN-DAG: "dynamic_work_items": 16
// VARIANCE-MEAN-DAG: "dataflow.load": 16
// VARIANCE-MEAN-DAG: "arith.addf": 16
// VARIANCE-MEAN-DAG: "arith.mulf": 17
// VARIANCE-MEAN-DAG: "f32:-0.062500"

// VARIANCE-VAR-DAG: "kind": "dfg_sim_report"
// VARIANCE-VAR-DAG: "workload": "variance"
// VARIANCE-VAR-DAG: "graph": "g_t_variance_red_1_0"
// VARIANCE-VAR-DAG: "status": "pass"
// VARIANCE-VAR-DAG: "optimistic_cycles": 314
// VARIANCE-VAR-DAG: "wavefront_steps": 37
// VARIANCE-VAR-DAG: "event_count": 168
// VARIANCE-VAR-DAG: "dynamic_work_items": 16
// VARIANCE-VAR-DAG: "dataflow.load": 16
// VARIANCE-VAR-DAG: "arith.subf": 16
// VARIANCE-VAR-DAG: "llvm.intr.fmuladd": 16
// VARIANCE-VAR-DAG: "f32:4.214844"

// MATVEC-DAG: "kind": "dfg_sim_report"
// MATVEC-DAG: "workload": "matvec"
// MATVEC-DAG: "graph": "g_t_matvec_kernel_0_0"
// MATVEC-DAG: "status": "pass"
// MATVEC-DAG: "optimistic_cycles": 83
// MATVEC-DAG: "wavefront_steps": 14
// MATVEC-DAG: "event_count": 43
// MATVEC-DAG: "dynamic_work_items": 5
// MATVEC-DAG: "arith.muli": 5
// MATVEC-DAG: "dataflow.load": 10
// MATVEC-DAG: "i32:55"

// MATVEC-ROW1-DAG: "kind": "dfg_sim_report"
// MATVEC-ROW1-DAG: "workload": "matvec"
// MATVEC-ROW1-DAG: "graph": "g_t_matvec_kernel_0_0"
// MATVEC-ROW1-DAG: "status": "pass"
// MATVEC-ROW1-DAG: "optimistic_cycles": 83
// MATVEC-ROW1-DAG: "i32:130"

// MATVEC-CHECKSUM-DAG: "kind": "dfg_sim_report"
// MATVEC-CHECKSUM-DAG: "workload": "matvec"
// MATVEC-CHECKSUM-DAG: "graph": "g_t_main_red_0_0"
// MATVEC-CHECKSUM-DAG: "status": "pass"
// MATVEC-CHECKSUM-DAG: "optimistic_cycles": 39
// MATVEC-CHECKSUM-DAG: "dynamic_work_items": 4
// MATVEC-CHECKSUM-DAG: "i32:370"

// VECADD-DAG: "kind": "dfg_sim_report"
// VECADD-DAG: "workload": "vecadd"
// VECADD-DAG: "graph": "g_t_vecadd_0_0"
// VECADD-DAG: "status": "pass"
// VECADD-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// VECADD-DAG: "optimistic_cycles": 960
// VECADD-DAG: "wavefront_steps": 67
// VECADD-DAG: "event_count": 320
// VECADD-DAG: "none"

// VECADD-REDUCTION-DAG: "kind": "dfg_sim_report"
// VECADD-REDUCTION-DAG: "workload": "vecadd"
// VECADD-REDUCTION-DAG: "graph": "g_t_main_red_0_0"
// VECADD-REDUCTION-DAG: "status": "pass"
// VECADD-REDUCTION-DAG: "optimistic_cycles": 643
// VECADD-REDUCTION-DAG: "f32:3024"

// VECMUL-DAG: "kind": "dfg_sim_report"
// VECMUL-DAG: "workload": "vecmul"
// VECMUL-DAG: "graph": "g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0"
// VECMUL-DAG: "status": "pass"
// VECMUL-DAG: "optimistic_cycles": 256
// VECMUL-DAG: "wavefront_steps": 19
// VECMUL-DAG: "event_count": 80
// VECMUL-DAG: "dynamic_work_items": 16
// VECMUL-DAG: "arith.mulf": 16

// VECSCALE-DAG: "kind": "dfg_sim_report"
// VECSCALE-DAG: "workload": "vecscale"
// VECSCALE-DAG: "graph": "g_t__ZN12_GLOBAL__N_118vecscale_candidateEPKjjPjj_0_0"
// VECSCALE-DAG: "status": "pass"
// VECSCALE-DAG: "optimistic_cycles": 384
// VECSCALE-DAG: "wavefront_steps": 35
// VECSCALE-DAG: "event_count": 128
// VECSCALE-DAG: "dynamic_work_items": 32
// VECSCALE-DAG: "arith.muli": 32

// VECSUM-DAG: "kind": "dfg_sim_report"
// VECSUM-DAG: "workload": "vecsum"
// VECSUM-DAG: "graph": "g_t_vecsum_red_0_0"
// VECSUM-DAG: "status": "pass"
// VECSUM-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// VECSUM-DAG: "optimistic_cycles": 579
// VECSUM-DAG: "wavefront_steps": 131
// VECSUM-DAG: "event_count": 387
// VECSUM-DAG: "dynamic_work_items": 64
// VECSUM-DAG: "i32:2116"

// REDUCTION-DAG: "kind": "dfg_sim_report"
// REDUCTION-DAG: "workload": "reduction"
// REDUCTION-DAG: "graph": "g_t_reduce_sum_red_0_0"
// REDUCTION-DAG: "status": "pass"
// REDUCTION-DAG: "optimistic_cycles": 1155
// REDUCTION-DAG: "wavefront_steps": 259
// REDUCTION-DAG: "event_count": 771
// REDUCTION-DAG: "dynamic_work_items": 128
// REDUCTION-DAG: "i32:8128"

// SPMV-DAG: "kind": "dfg_sim_report"
// SPMV-DAG: "workload": "spmv"
// SPMV-DAG: "graph": "g_t_spmv_kernel_red_0_0"
// SPMV-DAG: "status": "pass"
// SPMV-DAG: "optimistic_cycles": 47
// SPMV-DAG: "wavefront_steps": 11
// SPMV-DAG: "event_count": 25
// SPMV-DAG: "i32:12"

// MEAN-DAG: "kind": "dfg_sim_report"
// MEAN-DAG: "workload": "mean"
// MEAN-DAG: "graph": "g_t_mean_kernel_red_0_0"
// MEAN-DAG: "status": "pass"
// MEAN-DAG: "optimistic_cycles": 904
// MEAN-DAG: "event_count": 518
// MEAN-DAG: "f32:4.312500"

// DOTPRODUCT-DAG: "kind": "dfg_sim_report"
// DOTPRODUCT-DAG: "workload": "dotproduct"
// DOTPRODUCT-DAG: "graph": "g_t_dotproduct_red_0_0"
// DOTPRODUCT-DAG: "status": "pass"
// DOTPRODUCT-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// DOTPRODUCT-DAG: "optimistic_cycles": 1027
// DOTPRODUCT-DAG: "wavefront_steps": 131
// DOTPRODUCT-DAG: "event_count": 451
// DOTPRODUCT-DAG: "f32:2016"

// VECNORM-L1-DAG: "kind": "dfg_sim_report"
// VECNORM-L1-DAG: "workload": "vecnorm_l1"
// VECNORM-L1-DAG: "graph": "g_t_vecnorm_l1_red_0_0"
// VECNORM-L1-DAG: "status": "pass"
// VECNORM-L1-DAG: "optimistic_cycles": 643
// VECNORM-L1-DAG: "event_count": 451
// VECNORM-L1-DAG: "i32:171"

// VECNORM-L2-DAG: "kind": "dfg_sim_report"
// VECNORM-L2-DAG: "workload": "vecnorm_l2"
// VECNORM-L2-DAG: "graph": "g_t_vecnorm_l2_red_0_0"
// VECNORM-L2-DAG: "status": "pass"
// VECNORM-L2-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// VECNORM-L2-DAG: "optimistic_cycles": 771
// VECNORM-L2-DAG: "wavefront_steps": 132
// VECNORM-L2-DAG: "event_count": 451
// VECNORM-L2-DAG: "i32:619"

// PREFIX-SUM-DAG: "kind": "dfg_sim_report"
// PREFIX-SUM-DAG: "workload": "prefix_sum"
// PREFIX-SUM-DAG: "graph": "g_t_prefix_sum_red_0_0"
// PREFIX-SUM-DAG: "status": "pass"
// PREFIX-SUM-DAG: "optimistic_cycles": 835
// PREFIX-SUM-DAG: "event_count": 451
// PREFIX-SUM-DAG: "i32:2016"

// INTEGRATE-TRAPZ-DAG: "kind": "dfg_sim_report"
// INTEGRATE-TRAPZ-DAG: "workload": "integrate_trapz"
// INTEGRATE-TRAPZ-DAG: "graph": "g_t_integrate_trapz_red_0_0"
// INTEGRATE-TRAPZ-DAG: "status": "pass"
// INTEGRATE-TRAPZ-DAG: "metric_definition": "optimistic_pipeline_latency_throughput_sum"
// INTEGRATE-TRAPZ-DAG: "optimistic_cycles": 299
// INTEGRATE-TRAPZ-DAG: "wavefront_steps": 21
// INTEGRATE-TRAPZ-DAG: "event_count": 147
// INTEGRATE-TRAPZ-DAG: "f32:0.335938"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-DAG: axpy,136,,blocked,DFG-sim report available
// SUMMARY-DAG: bit_reverse,267,,blocked,DFG-sim report available
// SUMMARY-DAG: conv1d,83,,blocked,DFG-sim report available
// SUMMARY-DAG: convolve_1d,157,,blocked,DFG-sim report available
// SUMMARY-DAG: correlation,346,,blocked,DFG-sim report available
// SUMMARY-DAG: cumsum,14339,,blocked,DFG-sim report available
// SUMMARY-DAG: compare_swap,336,,blocked,DFG-sim report available
// SUMMARY-DAG: dotproduct,1027,,blocked,DFG-sim report available
// SUMMARY-DAG: hash_mix,1280,,blocked,DFG-sim report available
// SUMMARY-DAG: integrate_trapz,299,,blocked,DFG-sim report available
// SUMMARY-DAG: mean,904,,blocked,DFG-sim report available
// SUMMARY-DAG: matvec,371,,blocked,DFG-sim report available
// SUMMARY-DAG: prefix_sum,835,,blocked,DFG-sim report available
// SUMMARY-DAG: reduction,1155,,blocked,DFG-sim report available
// SUMMARY-DAG: relu,707,,blocked,DFG-sim report available
// SUMMARY-DAG: rotate_bits,544,,blocked,DFG-sim report available
// SUMMARY-DAG: spmv,47,,blocked,DFG-sim report available
// SUMMARY-DAG: variance,546,,blocked,DFG-sim report available
// SUMMARY-DAG: vecadd,1603,,blocked,DFG-sim report available
// SUMMARY-DAG: vecmul,256,,blocked,DFG-sim report available
// SUMMARY-DAG: vecnorm_l1,643,,blocked,DFG-sim report available
// SUMMARY-DAG: vecnorm_l2,771,,blocked,DFG-sim report available
// SUMMARY-DAG: vecscale,384,,blocked,DFG-sim report available
// SUMMARY-DAG: vecsum,579,,blocked,DFG-sim report available
// SUMMARY-DAG: xor_block,448,,blocked,DFG-sim report available
