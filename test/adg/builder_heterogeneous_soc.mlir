// RUN: loom-adg-builder-test --heterogeneous-soc --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: bash %S/../fabric/run_adg_hardware_summary.sh --input %t.hardware.mlir --input-recipe-identity %t.hardware.mlir=adg-builder::heterogeneous-soc --output %t.hardware.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.hardware.csv

// HARDWARE-LABEL: fabric.module @shared_reduction_adg
// HARDWARE-LABEL: fabric.system @heterogeneous_dual_accel_soc
// HARDWARE-SAME: memory_model = "sequential"
// HARDWARE: fabric.node @host0 kind = "host_core"
// HARDWARE: fabric.node @acc0 kind = "acc_core"
// HARDWARE: spatial = @shared_reduction_adg
// HARDWARE: fabric.node @fft0 kind = "fixed_accelerator"
// HARDWARE: fabric.node @l1d0 kind = "cache"
// HARDWARE: fabric.node @dram0 kind = "memory"
// HARDWARE: fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @l1d0 dst_port = "host" dst_channel = "aw"
// HARDWARE: fabric.link src = @l1d0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "cache" dst_channel = "aw"
// HARDWARE: fabric.link src = @acc0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "acc0" dst_channel = "aw"

// SUMMARY: {{.*}}::heterogeneous_dual_accel_soc,fabric_system,5,20,pass,fabric.system verified; link_count counts explicit fabric.link records,,,adg-builder::heterogeneous-soc,acc_core;cache;fixed_accelerator;host_core;memory
