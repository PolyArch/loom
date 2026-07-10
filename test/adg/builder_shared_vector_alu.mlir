// RUN: loom-adg-builder-test --shared-vector-alu --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @shared_vector_alu_adg
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: fabric.op [@arith.xori]
// HARDWARE-DAG: fabric.op [@llvm.intr.bswap]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@arith.muli]
// HARDWARE-DAG: fabric.op [@arith.addi]
// HARDWARE-DAG: fabric.op [@llvm.arm.qsub16]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
