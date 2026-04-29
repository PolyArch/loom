// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Pure hardware: bypassable = false (no software param possible).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @fifo_hw_not_bypassable
fabric.module @fifo_hw_not_bypassable(%a : !fabric.bits<8>) {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 4, bypassable = false] : !fabric.bits<8>
  %0 = fabric.fifo %a [max_depth = 4, bypassable = false] : !fabric.bits<8>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Pure hardware: bypassable = true, software param unset.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @fifo_hw_bypassable
fabric.module @fifo_hw_bypassable(%a : !fabric.bits<16>) {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 1, bypassable = true] : !fabric.bits<16>
  %0 = fabric.fifo %a [max_depth = 1, bypassable = true] : !fabric.bits<16>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: bypassable = true, bypassed = false.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @fifo_programmed_not_bypassed
fabric.module @fifo_programmed_not_bypassed(%a : !fabric.bits<8>) {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 8, bypassable = true] {bypassed = false} : !fabric.bits<8>
  %0 = fabric.fifo %a [max_depth = 8, bypassable = true] {bypassed = false} : !fabric.bits<8>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Programmed: bypassable = true, bypassed = true.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @fifo_programmed_bypassed
fabric.module @fifo_programmed_bypassed(%a : !fabric.bits<32>) {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 2, bypassable = true] {bypassed = true} : !fabric.bits<32>
  %0 = fabric.fifo %a [max_depth = 2, bypassable = true] {bypassed = true} : !fabric.bits<32>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Type variants: bits<0>, bits_tag, tag.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @fifo_bits_zero
fabric.module @fifo_bits_zero(%a : !fabric.bits<0>) {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 1, bypassable = false] : !fabric.bits<0>
  %0 = fabric.fifo %a [max_depth = 1, bypassable = false] : !fabric.bits<0>
  fabric.yield
}

// CHECK-LABEL: fabric.module @fifo_bits_tag
fabric.module @fifo_bits_tag(%a : !fabric.bits_tag<8, 2>) {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 4, bypassable = true] : !fabric.bits_tag<8, 2>
  %0 = fabric.fifo %a [max_depth = 4, bypassable = true] : !fabric.bits_tag<8, 2>
  fabric.yield
}

// CHECK-LABEL: fabric.module @fifo_tag
fabric.module @fifo_tag(%a : !fabric.tag<3>) {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 2, bypassable = false] : !fabric.tag<3>
  %0 = fabric.fifo %a [max_depth = 2, bypassable = false] : !fabric.tag<3>
  fabric.yield
}
