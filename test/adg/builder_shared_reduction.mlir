// RUN: rm -rf %t.dir
// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l1 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l1/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l2 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l2/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/downsample LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/downsample/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/matvec LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/matvec/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/gemv LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/gemv/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/dot_product_3d LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/dot_product_3d/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/spmv LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/spmv/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/sbox_lookup LOOM_CC=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/sbox_lookup/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/rotate_bits LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/rotate_bits/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/variance LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/variance/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l1/main_func.dfg.mlir --graph g_t_vecnorm_l1_red_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecnorm_l1 --output %t.dir/vecnorm_l1.mapping.csv --artifact %t.dir/vecnorm_l1.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l2/main_func.dfg.mlir --graph g_t_vecnorm_l2_red_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecnorm_l2 --output %t.dir/vecnorm_l2.mapping.csv --artifact %t.dir/vecnorm_l2.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/downsample/main_func.dfg.mlir --graph g_t_downsample_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload downsample --output %t.dir/downsample.mapping.csv --artifact %t.dir/downsample.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_matvec_kernel_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.mapping.csv --artifact %t.dir/matvec.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/gemv/main_func.dfg.mlir --graph g_t_gemv_kernel_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload gemv --output %t.dir/gemv.mapping.csv --artifact %t.dir/gemv.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/dot_product_3d/main_func.dfg.mlir --graph g_t_dot_product_3d_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload dot_product_3d --output %t.dir/dot_product_3d.mapping.csv --artifact %t.dir/dot_product_3d.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_vecadd_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/vecadd.mapping.csv --artifact %t.dir/vecadd.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/spmv/main_func.dfg.mlir --graph g_t_spmv_kernel_red_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload spmv --output %t.dir/spmv.mapping.csv --artifact %t.dir/spmv.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/sbox_lookup/main_func.dfg.mlir --graph g_t_main_2_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload sbox_lookup --output %t.dir/sbox_lookup.mapping.csv --artifact %t.dir/sbox_lookup.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/rotate_bits/main_func.dfg.mlir --graph g_t_rotate_bits_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload rotate_bits --output %t.dir/rotate_bits.mapping.csv --artifact %t.dir/rotate_bits.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/variance/main_func.dfg.mlir --graph g_t_variance_red_1_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload variance --output %t.dir/variance.mapping.csv --artifact %t.dir/variance.mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=VECNORM-L1 < %t.dir/vecnorm_l1.mapping.json
// RUN: FileCheck %s --check-prefix=VECNORM-L2 < %t.dir/vecnorm_l2.mapping.json
// RUN: FileCheck %s --check-prefix=DOWNSAMPLE < %t.dir/downsample.mapping.json
// RUN: FileCheck %s --check-prefix=MATVEC < %t.dir/matvec.mapping.json
// RUN: FileCheck %s --check-prefix=GEMV < %t.dir/gemv.mapping.json
// RUN: FileCheck %s --check-prefix=DOT3D < %t.dir/dot_product_3d.mapping.json
// RUN: FileCheck %s --check-prefix=VECADD < %t.dir/vecadd.mapping.json
// RUN: FileCheck %s --check-prefix=SPMV < %t.dir/spmv.mapping.json
// RUN: FileCheck %s --check-prefix=SBOX < %t.dir/sbox_lookup.mapping.json
// RUN: FileCheck %s --check-prefix=ROTATE-BITS < %t.dir/rotate_bits.mapping.json
// RUN: FileCheck %s --check-prefix=VARIANCE < %t.dir/variance.mapping.json

// HARDWARE-LABEL: fabric.module @shared_reduction_adg
// HARDWARE-DAG: fabric.op [@dataflow.stream]
// HARDWARE-DAG: fabric.op [@dataflow.carry]
// HARDWARE-DAG: fabric.op [@dataflow.invariant]
// HARDWARE-DAG: fabric.op [@arith.addi]
// HARDWARE-DAG: fabric.op [@arith.addi, @arith.subi]
// HARDWARE-DAG: fabric.op [@llvm.intr.abs]
// HARDWARE-DAG: fabric.op [@llvm.intr.fabs]
// HARDWARE-DAG: fabric.op [@arith.muli]
// HARDWARE-DAG: fabric.op [@arith.addf]
// HARDWARE-DAG: fabric.op [@arith.subf]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@arith.shrui]
// HARDWARE-DAG: fabric.op [@arith.shli]
// HARDWARE-DAG: fabric.op [@arith.andi]
// HARDWARE-DAG: fabric.op [@arith.ori]
// HARDWARE-DAG: fabric.op [@llvm.arm.qadd16, @llvm.arm.qsub16]
// HARDWARE-DAG: fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-DAG: fabric.mem [spatial]

// MAPPING-DAG: "hardware": "shared_reduction_adg"
// MAPPING-DAG: "placed_records": 5
// MAPPING-DAG: "routed_edges": 6
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "config_records": 137
// MAPPING-DAG: "status": "pass"

// VECNORM-L1-DAG: "workload": "vecnorm_l1"
// VECNORM-L1-DAG: "hardware": "shared_reduction_adg"
// VECNORM-L1-DAG: "placed_records": 6
// VECNORM-L1-DAG: "routed_edges": {{[1-9][0-9]*}}
// VECNORM-L1-DAG: "unrouted_edges": 0
// VECNORM-L1-DAG: "status": "pass"
// VECNORM-L1-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.abs#0.operand0"
// VECNORM-L1-DAG: "edge_ref": "llvm.intr.abs#0.result0->arith.addi#0.operand0"

// VECNORM-L2-DAG: "workload": "vecnorm_l2"
// VECNORM-L2-DAG: "hardware": "shared_reduction_adg"
// VECNORM-L2-DAG: "placed_records": 6
// VECNORM-L2-DAG: "routed_edges": {{[1-9][0-9]*}}
// VECNORM-L2-DAG: "unrouted_edges": 0
// VECNORM-L2-DAG: "status": "pass"
// VECNORM-L2-DAG: "edge_ref": "dataflow.load#0.result0->arith.muli#0.operand0"
// VECNORM-L2-DAG: "edge_ref": "dataflow.load#0.result0->arith.muli#0.operand1"
// VECNORM-L2-DAG: "edge_ref": "arith.muli#0.result0->arith.addi#0.operand0"

// DOWNSAMPLE-DAG: "workload": "downsample"
// DOWNSAMPLE-DAG: "hardware": "shared_reduction_adg"
// DOWNSAMPLE-DAG: "placed_records": 6
// DOWNSAMPLE-DAG: "routed_edges": 6
// DOWNSAMPLE-DAG: "unrouted_edges": 0
// DOWNSAMPLE-DAG: "status": "pass"
// DOWNSAMPLE-DAG: "edge_ref": "dataflow.constant#0.result0->arith.shrui#0.operand1"
// DOWNSAMPLE-DAG: "edge_ref": "arith.shli#0.result0->arith.shrui#0.operand0"
// DOWNSAMPLE-DAG: "edge_ref": "arith.shrui#0.result0->dataflow.load#0.operand1"
// DOWNSAMPLE-DAG: "edge_ref": "dataflow.load#0.result0->dataflow.store#0.operand2"
// DOWNSAMPLE-DAG: "segment_kind": "module_path"
// DOWNSAMPLE-NOT: ".out"
// DOWNSAMPLE-NOT: ".in"

// MATVEC-DAG: "workload": "matvec"
// MATVEC-DAG: "hardware": "shared_reduction_adg"
// MATVEC-DAG: "placed_records": 7
// MATVEC-DAG: "routed_edges": {{[1-9][0-9]*}}
// MATVEC-DAG: "unrouted_edges": 0
// MATVEC-DAG: "status": "pass"
// MATVEC-DAG: "edge_ref": "dataflow.load#1.result0->arith.muli#0.operand0"
// MATVEC-DAG: "edge_ref": "dataflow.load#1.result1->dataflow.sync#0.operand1"
// MATVEC-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#1.operand1"

// GEMV-DAG: "workload": "gemv"
// GEMV-DAG: "hardware": "shared_reduction_adg"
// GEMV-DAG: "placed_records": 9
// GEMV-DAG: "routed_edges": {{[1-9][0-9]*}}
// GEMV-DAG: "unrouted_edges": 0
// GEMV-DAG: "status": "pass"
// GEMV-DAG: "edge_ref": "dataflow.carry#0.result0->arith.shli#0.operand0"
// GEMV-DAG: "edge_ref": "dataflow.invariant#0.result0->arith.shli#0.operand1"
// GEMV-DAG: "segment_kind": "module_path"
// GEMV-NOT: ".out"
// GEMV-NOT: ".in"

// DOT3D-DAG: "workload": "dot_product_3d"
// DOT3D-DAG: "hardware": "shared_reduction_adg"
// DOT3D-DAG: "placed_records": 14
// DOT3D-DAG: "unrouted_edges": 0
// DOT3D-DAG: "status": "pass"
// DOT3D-DAG: "edge_ref": "llvm.intr.fmuladd#0.result0->llvm.intr.fmuladd#1.operand2"
// DOT3D-DAG: "edge_ref": "llvm.intr.fmuladd#1.result0->dataflow.store#0.operand2"
// DOT3D-DAG: "segment_kind": "module_path"
// DOT3D-NOT: ".out"
// DOT3D-NOT: ".in"

// VECADD-DAG: "workload": "vecadd"
// VECADD-DAG: "hardware": "shared_reduction_adg"
// VECADD-DAG: "placed_records": 5
// VECADD-DAG: "routed_edges": {{[1-9][0-9]*}}
// VECADD-DAG: "unrouted_edges": 0
// VECADD-DAG: "status": "pass"
// VECADD-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// VECADD-DAG: "edge_ref": "dataflow.load#1.result0->arith.addf#0.operand1"
// VECADD-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand2"

// SPMV-DAG: "workload": "spmv"
// SPMV-DAG: "hardware": "shared_reduction_adg"
// SPMV-DAG: "placed_records": 9
// SPMV-DAG: "routed_edges": {{[1-9][0-9]*}}
// SPMV-DAG: "unrouted_edges": 0
// SPMV-DAG: "status": "pass"
// SPMV-DAG: "edge_ref": "dataflow.load#1.result0->llvm.zext#0.operand0"
// SPMV-DAG: "edge_ref": "llvm.zext#0.result0->dataflow.load#2.operand1"
// SPMV-DAG: "edge_ref": "dataflow.load#2.result1->dataflow.sync#0.operand2"
// SPMV-DAG: "segment_kind": "module_path"
// SPMV-NOT: ".out"
// SPMV-NOT: ".in"

// SBOX-DAG: "workload": "sbox_lookup"
// SBOX-DAG: "hardware": "shared_reduction_adg"
// SBOX-DAG: "placed_records": 6
// SBOX-DAG: "routed_edges": 7
// SBOX-DAG: "unrouted_edges": 0
// SBOX-DAG: "status": "pass"
// SBOX-DAG: "edge_ref": "dataflow.load#0.result0->arith.andi#0.operand0"
// SBOX-DAG: "edge_ref": "arith.andi#0.result0->llvm.zext#0.operand0"
// SBOX-DAG: "edge_ref": "llvm.zext#0.result0->dataflow.load#1.operand1"
// SBOX-DAG: "edge_ref": "dataflow.load#1.result0->dataflow.store#0.operand2"
// SBOX-DAG: "segment_kind": "module_path"
// SBOX-NOT: ".out"
// SBOX-NOT: ".in"

// ROTATE-BITS-DAG: "workload": "rotate_bits"
// ROTATE-BITS-DAG: "hardware": "shared_reduction_adg"
// ROTATE-BITS-DAG: "placed_records": 8
// ROTATE-BITS-DAG: "routed_edges": 12
// ROTATE-BITS-DAG: "unrouted_edges": 0
// ROTATE-BITS-DAG: "status": "pass"
// ROTATE-BITS-DAG: "edge_ref": "dataflow.load#1.result0->llvm.intr.fshl#0.operand0"
// ROTATE-BITS-DAG: "edge_ref": "arith.cmpi#0.result0->arith.select#0.operand0"
// ROTATE-BITS-DAG: "edge_ref": "llvm.intr.fshl#0.result0->arith.select#0.operand2"
// ROTATE-BITS-DAG: "edge_ref": "arith.select#0.result0->dataflow.store#0.operand2"
// ROTATE-BITS-DAG: "segment_kind": "module_path"
// ROTATE-BITS-NOT: ".out"
// ROTATE-BITS-NOT: ".in"

// VARIANCE-DAG: "workload": "variance"
// VARIANCE-DAG: "hardware": "shared_reduction_adg"
// VARIANCE-DAG: "placed_records": 9
// VARIANCE-DAG: "routed_edges": {{[1-9][0-9]*}}
// VARIANCE-DAG: "unrouted_edges": 0
// VARIANCE-DAG: "status": "pass"
// VARIANCE-DAG: "edge_ref": "dataflow.load#0.result0->arith.subf#0.operand0"
// VARIANCE-DAG: "edge_ref": "dataflow.invariant#1.result0->arith.subf#0.operand1"
// VARIANCE-DAG: "edge_ref": "arith.subf#0.result0->llvm.intr.fmuladd#0.operand0"
// VARIANCE-DAG: "edge_ref": "arith.subf#0.result0->llvm.intr.fmuladd#0.operand1"
// VARIANCE-DAG: "edge_ref": "dataflow.stream#0.result1->dataflow.invariant#1.operand0"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::mem.load#0.result0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.operand1"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.result0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand0"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#{{[0-9]+}}.result0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.operand1"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.result0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand1"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#{{[0-9]+}}.result0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.operand4"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.result0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.operand4"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::fabric.switch#{{[0-9]+}}.result0"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand1"
// VARIANCE-DAG: "source_endpoint": "shared_reduction_adg::fabric.pe#{{[0-9]+}}.result4"
// VARIANCE-DAG: "sink_endpoint": "shared_reduction_adg::fabric.op#{{[0-9]+}}.operand0"
// VARIANCE-DAG: "segment_kind": "module_path"
// VARIANCE-NOT: ".out"
// VARIANCE-NOT: ".in"
