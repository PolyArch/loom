// RUN: loom-adg-builder-test --shared-reduction --output %t.hardware.mlir
// RUN: FileCheck %s --check-prefix=BUILDER < %t.hardware.mlir
// RUN: sed -n '/^fabric.module/,$p' %S/../pnr/shared_reduction_adg.mlir > %t.fixture.mlir
// RUN: diff %t.fixture.mlir %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @shared_reduction_adg
// HARDWARE-DAG: fabric.switch [spatial] %arg8 [{connectivity_table = ["1", "1"
// HARDWARE-DAG: fabric.op [@dataflow.stream]
// HARDWARE-DAG: cont_cond = ["<", ">"]
// HARDWARE-DAG: fabric.op [@dataflow.carry]
// HARDWARE-DAG: fabric.op [@dataflow.invariant]
// HARDWARE-DAG: fabric.op [@dataflow.gate]
// HARDWARE-DAG: fabric.op [@dataflow.demux]
// HARDWARE-DAG: fabric.op [@dataflow.mux]
// HARDWARE-DAG: fabric.op [@arith.addi]
// HARDWARE-DAG: fabric.op [@arith.addi, @arith.subi]
// HARDWARE-DAG: fabric.op [@arith.divsi]
// HARDWARE-DAG: fabric.op [@arith.remsi]
// HARDWARE-DAG: fabric.op [@arith.divui, @arith.remui]
// HARDWARE-DAG: fabric.op [@llvm.intr.abs]
// HARDWARE-DAG: fabric.op [@llvm.intr.fabs]
// HARDWARE-DAG: fabric.op [@llvm.intr.umax]
// HARDWARE-DAG: fabric.op [@llvm.intr.umin]
// HARDWARE-DAG: fabric.op [@arith.muli]
// HARDWARE-DAG: fabric.op [@arith.addf]
// HARDWARE-DAG: fabric.op [@arith.subf]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@arith.divf, @arith.remf]
// HARDWARE-DAG: predicate = ["oeq", "ogt", "ugt", "ule", "olt"]
// HARDWARE-DAG: fabric.op [@arith.shrsi, @arith.shrui]
// HARDWARE-DAG: fabric.op [@arith.shli]
// HARDWARE-DAG: fabric.op [@arith.andi]
// HARDWARE-DAG: fabric.op [@arith.ori]
// HARDWARE-DAG: fabric.op [@llvm.arm.qadd16, @llvm.arm.sadd16, @llvm.arm.qsub16, @llvm.arm.qsub8]
// HARDWARE-DAG: fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext]
// HARDWARE-DAG: fabric.op [@arith.index_cast]
// HARDWARE-DAG: fabric.op [@llvm.sext, @llvm.zext]
// HARDWARE-DAG: fabric.op [@llvm.trunc]
// HARDWARE-DAG: fabric.op [@llvm.select]
// HARDWARE-DAG: fabric.fifo
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-DAG: fabric.mem [spatial]

// BUILDER-DAG: %aux_gate_cond1, %aux_active_idx1 = fabric.pe [spatial]
// BUILDER-DAG: %gate_value1 = fabric.switch [spatial]
// BUILDER-DAG: %bit_invariant_value = fabric.switch [spatial]
// BUILDER-DAG: %bit_invariant_aux0_value = fabric.switch [spatial]
