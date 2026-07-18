// RUN: loom-adg-builder-test --minimal-spatial --output %t.hardware.mlir
// RUN: diff %S/../pnr/minimal_spatial_adg.mlir.inc %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @minimal_spatial_adg
// HARDWARE-DAG: %[[LHS_FANOUT:[0-9]+]]:2 = fabric.switch [spatial] %arg1 [{connectivity_table = ["1", "1"]}] : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
// HARDWARE-DAG: %[[RHS_FANOUT:[0-9]+]]:2 = fabric.switch [spatial] %arg2 [{connectivity_table = ["1", "1"]}] : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
// HARDWARE-DAG: fabric.pe [spatial] ({{.*}}%[[LHS_FANOUT]]#0{{.*}}%[[RHS_FANOUT]]#0
// HARDWARE-DAG: fabric.switch [spatial] %[[LHS_FANOUT]]#1, %[[RHS_FANOUT]]#1 [{connectivity_table = ["11", "11"]}]
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: data_width = 32 : i32
