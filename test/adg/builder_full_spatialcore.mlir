// RUN: loom-adg-builder-test --full-spatialcore --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir > %t.canonical.mlir
// RUN: FileCheck %s --check-prefix=HARDWARE < %t.canonical.mlir
// RUN: FileCheck %s --check-prefix=DISPATCH < %t.canonical.mlir

// HARDWARE-LABEL: fabric.module @full_spatialcore_adg(
// HARDWARE-DAG: fabric.pe [spatial]
// HARDWARE-DAG: fabric.pe [temporal]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: fabric.switch [temporal]
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE: %[[MEM:[0-9]+]]:5 = fabric.mem [temporal] mgr(%{{[^,]+}}, %{{[^)]+}})
// HARDWARE-DAG: operation_table_size = 2 : i32
// HARDWARE-DAG: fabric.boundary [s2t]
// HARDWARE-DAG: fabric.fifo
// HARDWARE-DAG: fabric.pe @ALU
// HARDWARE-DAG: fabric.instantiate @ALU
// HARDWARE: fabric.yield %[[MEM]]#1 : memref<?x!fabric.bits<16>>

// DISPATCH: fabric.mem [temporal]
// DISPATCH: dispatch_eligibility = [
// DISPATCH: [0 : i32
// DISPATCH: 1 : i32]
// DISPATCH: [0 : i32
// DISPATCH: 1 : i32]
