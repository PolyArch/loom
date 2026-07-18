// RUN: loom-adg-builder-test --minimal-temporal --output %t.hardware.mlir
// RUN: diff %S/../pnr/minimal_temporal_adg.mlir.inc %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @minimal_temporal_adg
// HARDWARE-DAG: %[[LHS_FANOUT:[0-9]+]]:2 = fabric.switch [temporal] %arg1 [{connectivity_table = ["1", "1"], route_table_size = 16 : i32}] : (!fabric.bits_tag<32, 4>) -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
// HARDWARE-DAG: %[[RHS_FANOUT:[0-9]+]]:2 = fabric.switch [temporal] %arg2 [{connectivity_table = ["1", "1"], route_table_size = 16 : i32}] : (!fabric.bits_tag<32, 4>) -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
// HARDWARE-DAG: fabric.pe [temporal] ({{.*}}%[[LHS_FANOUT]]#0{{.*}}%[[RHS_FANOUT]]#0
// HARDWARE-DAG: tag_width = 4 : i32
// HARDWARE-DAG: operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>
// HARDWARE-DAG: fabric.switch [temporal] %[[LHS_FANOUT]]#1, %[[RHS_FANOUT]]#1
// HARDWARE-DAG: route_table_size = 1 : i32
// HARDWARE-DAG: fabric.mem [temporal]
// HARDWARE-DAG: data_width = 32 : i32
// HARDWARE-DAG: operation_table_size = 1 : i32
// HARDWARE-DAG: dispatch_eligibility = {
// HARDWARE-DAG: operation_port_requests = {{\[\[0 : i32\]\]}}
// HARDWARE-DAG: subordinate_requests = {{\[\]}}
