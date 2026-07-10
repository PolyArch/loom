// RUN: loom-adg-builder-test --heterogeneous-soc --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// HARDWARE-LABEL: fabric.module @shared_reduction_adg
// HARDWARE-LABEL: fabric.system @heterogeneous_dual_accel_soc
// HARDWARE-SAME: memory_model = "sequential"
// HARDWARE: fabric.node @host0 kind = "host_core"
// HARDWARE: fabric.node @acc0 kind = "acc_core"
// HARDWARE-SAME: spatial = @shared_reduction_adg
// HARDWARE: fabric.node @fft0 kind = "fixed_accelerator"
// HARDWARE: fabric.node @l1d0 kind = "cache"
// HARDWARE: fabric.node @dma0 kind = "dma_engine"
// HARDWARE: fabric.node @dram0 kind = "memory"
// HARDWARE: fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @l1d0 dst_port = "host" dst_channel = "aw"
// HARDWARE: fabric.link src = @l1d0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "cache" dst_channel = "aw"
// HARDWARE: fabric.link src = @acc0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "acc0" dst_channel = "aw"
