// RUN: loom-adg-builder-test --minimal-spatial --output %t.hardware.mlir
// RUN: diff %S/../pnr/minimal_spatial_adg.mlir.inc %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @minimal_spatial_adg
// HARDWARE: fabric.pe [spatial]
// HARDWARE: fabric.switch [spatial]
// HARDWARE-SAME: connectivity_table = ["11", "11"]
// HARDWARE: fabric.mem [spatial]
