// RUN: loom-adg-builder-test --full-spatialcore --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: bash %S/../fabric/run_adg_hardware_summary.sh --input %t.hardware.mlir --input-recipe-identity %t.hardware.mlir=adg-builder::full-spatialcore --output %t.hardware.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.hardware.csv

// HARDWARE-LABEL: fabric.module @full_spatialcore_adg
// HARDWARE-DAG: fabric.pe [spatial]
// HARDWARE-DAG: fabric.pe [temporal]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: fabric.switch [temporal]
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: fabric.mem [temporal]
// HARDWARE-DAG: fabric.boundary [s2t]
// HARDWARE-DAG: fabric.fifo
// HARDWARE-DAG: fabric.pe @ALU
// HARDWARE-DAG: fabric.instantiate @ALU

// SUMMARY: {{.*}}::full_spatialcore_adg,fabric_module_template,10,0,pass,fabric.module template verified; link_count counts explicit fabric.link records only,mem;pe;switch,spatial;temporal,adg-builder::full-spatialcore,
