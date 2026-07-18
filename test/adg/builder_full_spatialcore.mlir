// RUN: loom-adg-builder-test --full-spatialcore --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @full_spatialcore_adg
// HARDWARE-DAG: fabric.pe [spatial]
// HARDWARE-DAG: fabric.pe [temporal]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: fabric.switch [temporal]
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: fabric.mem [temporal]
// HARDWARE-DAG: operation_table_size = 2 : i32
// HARDWARE-DAG: {{dispatch_eligibility = \[\[0 : i32, 1 : i32\], \[0 : i32, 1 : i32\]\]}}
// HARDWARE-DAG: fabric.boundary [s2t]
// HARDWARE-DAG: fabric.fifo
// HARDWARE-DAG: fabric.pe @ALU
// HARDWARE-DAG: fabric.instantiate @ALU
