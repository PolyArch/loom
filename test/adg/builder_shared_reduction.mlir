// RUN: rm -rf %t.dir
// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: env BUILD_DIR=%t.dir/vecsum LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecsum/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l1 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l1/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecnorm_l2 LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecnorm_l2/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/matvec LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/matvec/dfg_check.sh
// RUN: env BUILD_DIR=%t.dir/vecadd LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/vecadd/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecsum/main_func.dfg.mlir --graph g_t_vecsum_red_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecsum --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l1/main_func.dfg.mlir --graph g_t_vecnorm_l1_red_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecnorm_l1 --output %t.dir/vecnorm_l1.mapping.csv --artifact %t.dir/vecnorm_l1.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecnorm_l2/main_func.dfg.mlir --graph g_t_vecnorm_l2_red_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecnorm_l2 --output %t.dir/vecnorm_l2.mapping.csv --artifact %t.dir/vecnorm_l2.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/matvec/main_func.dfg.mlir --graph g_t_matvec_kernel_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload matvec --output %t.dir/matvec.mapping.csv --artifact %t.dir/matvec.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/vecadd/main_func.dfg.mlir --graph g_t_vecadd_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_reduction_adg --workload vecadd --output %t.dir/vecadd.mapping.csv --artifact %t.dir/vecadd.mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=VECNORM-L1 < %t.dir/vecnorm_l1.mapping.json
// RUN: FileCheck %s --check-prefix=VECNORM-L2 < %t.dir/vecnorm_l2.mapping.json
// RUN: FileCheck %s --check-prefix=MATVEC < %t.dir/matvec.mapping.json
// RUN: FileCheck %s --check-prefix=VECADD < %t.dir/vecadd.mapping.json

// HARDWARE-LABEL: fabric.module @shared_reduction_adg
// HARDWARE-DAG: fabric.op [@dataflow.stream]
// HARDWARE-DAG: fabric.op [@dataflow.carry]
// HARDWARE-DAG: fabric.op [@dataflow.invariant]
// HARDWARE-DAG: fabric.op [@arith.addi]
// HARDWARE-DAG: fabric.op [@llvm.intr.abs]
// HARDWARE-DAG: fabric.op [@arith.muli]
// HARDWARE-DAG: fabric.op [@arith.addf]
// HARDWARE-DAG: fabric.op [@arith.subf]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@arith.shrui]
// HARDWARE-DAG: fabric.op [@arith.shli]
// HARDWARE-DAG: fabric.op [@arith.andi]
// HARDWARE-DAG: fabric.op [@arith.ori]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-DAG: fabric.mem [spatial]

// MAPPING-DAG: "hardware": "shared_reduction_adg"
// MAPPING-DAG: "placed_records": 5
// MAPPING-DAG: "routed_edges": 6
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "config_records": 97
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

// MATVEC-DAG: "workload": "matvec"
// MATVEC-DAG: "hardware": "shared_reduction_adg"
// MATVEC-DAG: "placed_records": 7
// MATVEC-DAG: "routed_edges": {{[1-9][0-9]*}}
// MATVEC-DAG: "unrouted_edges": 0
// MATVEC-DAG: "status": "pass"
// MATVEC-DAG: "edge_ref": "dataflow.load#1.result0->arith.muli#0.operand0"
// MATVEC-DAG: "edge_ref": "dataflow.load#1.result1->dataflow.sync#0.operand1"
// MATVEC-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#1.operand1"

// VECADD-DAG: "workload": "vecadd"
// VECADD-DAG: "hardware": "shared_reduction_adg"
// VECADD-DAG: "placed_records": 5
// VECADD-DAG: "routed_edges": {{[1-9][0-9]*}}
// VECADD-DAG: "unrouted_edges": 0
// VECADD-DAG: "status": "pass"
// VECADD-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// VECADD-DAG: "edge_ref": "dataflow.load#1.result0->arith.addf#0.operand1"
// VECADD-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand2"
