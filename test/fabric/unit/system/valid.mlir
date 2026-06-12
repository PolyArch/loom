// RUN: loom %s | loom | FileCheck %s

fabric.module @shared_reduction_adg(%mgr : memref<?x!fabric.bits<32>>) {
  fabric.yield
}

// CHECK-LABEL: fabric.system @heterogeneous_dual_accel_soc
// CHECK-SAME: memory_model = "sequential"
// CHECK: fabric.node @host0 kind = "host_core"
// CHECK: fabric.node @acc0 kind = "acc_core"
// CHECK: fabric.node @fft0 kind = "fixed_accelerator"
// CHECK: fabric.node @l1d0 kind = "cache"
// CHECK: fabric.node @dram0 kind = "memory"
// CHECK: fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @l1d0 dst_port = "host" dst_channel = "aw"
fabric.system @heterogeneous_dual_accel_soc memory_model = "sequential" {
  fabric.node @host0 kind = "host_core"
      ports = ["mem.aw:output", "mem.w:output", "mem.b:input", "mem.ar:output", "mem.r:input"]
      attributes {scalar = "rv64gc"}
  fabric.node @acc0 kind = "acc_core"
      ports = ["mem.aw:output", "mem.w:output", "mem.b:input", "mem.ar:output", "mem.r:input"]
      attributes {spatial = @shared_reduction_adg, scalar = "rv32im"}
  fabric.node @fft0 kind = "fixed_accelerator"
      ports = ["mem.aw:output", "mem.w:output", "mem.b:input", "mem.ar:output", "mem.r:input"]
      attributes {function = "fft"}
  fabric.node @l1d0 kind = "cache"
      ports = ["host.aw:input", "host.w:input", "host.b:output", "host.ar:input", "host.r:output",
               "mem.aw:output", "mem.w:output", "mem.b:input", "mem.ar:output", "mem.r:input"]
      attributes {params = {line_bytes = 64 : i64, capacity_bytes = 32768 : i64}}
  fabric.node @dram0 kind = "memory"
      ports = ["cache.aw:input", "cache.w:input", "cache.b:output", "cache.ar:input", "cache.r:output",
               "acc0.aw:input", "acc0.w:input", "acc0.b:output", "acc0.ar:input", "acc0.r:output",
               "fft0.aw:input", "fft0.w:input", "fft0.b:output", "fft0.ar:input", "fft0.r:output"]
      attributes {bytes = 1048576 : i64}

  fabric.link src = @host0 src_port = "mem" src_channel = "aw" dst = @l1d0 dst_port = "host" dst_channel = "aw"
  fabric.link src = @host0 src_port = "mem" src_channel = "w" dst = @l1d0 dst_port = "host" dst_channel = "w"
  fabric.link src = @l1d0 src_port = "host" src_channel = "b" dst = @host0 dst_port = "mem" dst_channel = "b"
  fabric.link src = @host0 src_port = "mem" src_channel = "ar" dst = @l1d0 dst_port = "host" dst_channel = "ar"
  fabric.link src = @l1d0 src_port = "host" src_channel = "r" dst = @host0 dst_port = "mem" dst_channel = "r"

  fabric.link src = @l1d0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "cache" dst_channel = "aw"
  fabric.link src = @l1d0 src_port = "mem" src_channel = "w" dst = @dram0 dst_port = "cache" dst_channel = "w"
  fabric.link src = @dram0 src_port = "cache" src_channel = "b" dst = @l1d0 dst_port = "mem" dst_channel = "b"
  fabric.link src = @l1d0 src_port = "mem" src_channel = "ar" dst = @dram0 dst_port = "cache" dst_channel = "ar"
  fabric.link src = @dram0 src_port = "cache" src_channel = "r" dst = @l1d0 dst_port = "mem" dst_channel = "r"

  fabric.link src = @acc0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "acc0" dst_channel = "aw"
  fabric.link src = @acc0 src_port = "mem" src_channel = "w" dst = @dram0 dst_port = "acc0" dst_channel = "w"
  fabric.link src = @dram0 src_port = "acc0" src_channel = "b" dst = @acc0 dst_port = "mem" dst_channel = "b"
  fabric.link src = @acc0 src_port = "mem" src_channel = "ar" dst = @dram0 dst_port = "acc0" dst_channel = "ar"
  fabric.link src = @dram0 src_port = "acc0" src_channel = "r" dst = @acc0 dst_port = "mem" dst_channel = "r"

  fabric.link src = @fft0 src_port = "mem" src_channel = "aw" dst = @dram0 dst_port = "fft0" dst_channel = "aw"
  fabric.link src = @fft0 src_port = "mem" src_channel = "w" dst = @dram0 dst_port = "fft0" dst_channel = "w"
  fabric.link src = @dram0 src_port = "fft0" src_channel = "b" dst = @fft0 dst_port = "mem" dst_channel = "b"
  fabric.link src = @fft0 src_port = "mem" src_channel = "ar" dst = @dram0 dst_port = "fft0" dst_channel = "ar"
  fabric.link src = @dram0 src_port = "fft0" src_channel = "r" dst = @fft0 dst_port = "mem" dst_channel = "r"
}
