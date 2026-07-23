// RUN: loom-adg-builder-test --full-spatialcore --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir > %t.canonical.mlir
// RUN: FileCheck %s --check-prefix=HARDWARE < %t.canonical.mlir
// RUN: FileCheck %s --check-prefix=DISPATCH < %t.canonical.mlir
// RUN: FileCheck %s --check-prefix=NAMEDPE < %t.canonical.mlir

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

// The named PE template's port signature is owned by `function_type`; its
// body therefore closes with a zero-operand signature terminator.
// NAMEDPE: fabric.pe @ALU [spatial] (!fabric.bits<32>) -> !fabric.bits<32>
// NAMEDPE: fabric.yield %{{.*}} : !fabric.bits<32>
// NAMEDPE-NEXT: }
// NAMEDPE-NEXT: fabric.yield{{[[:space:]]*$}}

// DISPATCH: fabric.mem [spatial]
// DISPATCH-DAG: operation_port_requests = {{\[\[0 : i32\], \[0 : i32\]\]}}
// DISPATCH-DAG: subordinate_requests = {{\[\]}}
// DISPATCH: fabric.mem [temporal]
// DISPATCH-DAG: operation_port_requests = {{\[\[0 : i32, 1 : i32\], \[0 : i32, 1 : i32\]\]}}
// DISPATCH-DAG: subordinate_requests = {{\[\[0 : i32, 1 : i32\], \[0 : i32, 1 : i32\]\]}}

// HARDWARE-LABEL: fabric.module @temporal_mem_capacity_anchors_adg(
// HARDWARE: fabric.mem [temporal]
// HARDWARE-DAG: load_group_size = 2 : i32
// HARDWARE-DAG: tag_width = 4 : i32
// HARDWARE-DAG: operation_table_size = 17 : i32

// HARDWARE: fabric.mem [temporal]
// HARDWARE-DAG: load_group_size = 1 : i32
// HARDWARE-DAG: tag_width = 64 : i32
// HARDWARE-DAG: operation_table_size = 2147483647 : i32
