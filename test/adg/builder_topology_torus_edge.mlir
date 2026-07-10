// RUN: loom-adg-builder-test --topology-matrix-case torus-edge --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @matrix_torus_edge_adg
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: fabric.pe [spatial]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: connectivity_table = ["110", "101"]
