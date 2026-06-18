// RUN: loom-adg-builder-test --topology-matrix-case torus-edge --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: bash %S/../fabric/run_adg_hardware_summary.sh --input %t.hardware.mlir --input-recipe-identity %t.hardware.mlir=adg-builder::topology-torus-edge --output %t.hardware.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.hardware.csv

// HARDWARE-LABEL: fabric.module @matrix_torus_edge_adg
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: fabric.pe [spatial]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: connectivity_table = ["110", "101"]

// SUMMARY: {{.*}}::matrix_torus_edge_adg,fabric_module_template,7,0,pass,fabric.module template verified; link_count counts explicit fabric.link records only,mem;pe;switch,spatial,adg-builder::topology-torus-edge,
