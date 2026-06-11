// RUN: loom-adg-builder-test --minimal-temporal --output %t.hardware.mlir
// RUN: diff %S/../pnr/minimal_temporal_adg.mlir.inc %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @minimal_temporal_adg
// HARDWARE: fabric.pe [temporal]
// HARDWARE-SAME: !fabric.bits_tag<32, 4>
// HARDWARE: tag_width = 4 : i32
// HARDWARE: fabric.switch [temporal]
// HARDWARE-SAME: route_table_size = 1 : i32
// HARDWARE: fabric.mem [temporal]
// HARDWARE-SAME: addr_table_size = 1 : i32
