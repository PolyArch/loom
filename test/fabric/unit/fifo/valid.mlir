// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Pure hardware: bypassable = false (no software param possible).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @fifo_hw_not_bypassable
func.func @fifo_hw_not_bypassable(%a: !fabric.bits<8>) -> !fabric.bits<8> {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 4, bypassable = false] : !fabric.bits<8>
  %0 = fabric.fifo %a [max_depth = 4, bypassable = false] : !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----------------------------------------------------------------------------
// Pure hardware: bypassable = true, software param unset.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @fifo_hw_bypassable
func.func @fifo_hw_bypassable(%a: !fabric.bits<16>) -> !fabric.bits<16> {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 1, bypassable = true] : !fabric.bits<16>
  %0 = fabric.fifo %a [max_depth = 1, bypassable = true] : !fabric.bits<16>
  return %0 : !fabric.bits<16>
}

// -----------------------------------------------------------------------------
// Programmed: bypassable = true, bypassed = false.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @fifo_programmed_not_bypassed
func.func @fifo_programmed_not_bypassed(%a: !fabric.bits<8>) -> !fabric.bits<8> {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 8, bypassable = true] {bypassed = false} : !fabric.bits<8>
  %0 = fabric.fifo %a [max_depth = 8, bypassable = true] {bypassed = false} : !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----------------------------------------------------------------------------
// Programmed: bypassable = true, bypassed = true.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @fifo_programmed_bypassed
func.func @fifo_programmed_bypassed(%a: !fabric.bits<32>) -> !fabric.bits<32> {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 2, bypassable = true] {bypassed = true} : !fabric.bits<32>
  %0 = fabric.fifo %a [max_depth = 2, bypassable = true] {bypassed = true} : !fabric.bits<32>
  return %0 : !fabric.bits<32>
}

// -----------------------------------------------------------------------------
// Type variants: bits<0>, bits_tag, tag.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @fifo_bits_zero
func.func @fifo_bits_zero(%a: !fabric.bits<0>) -> !fabric.bits<0> {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 1, bypassable = false] : !fabric.bits<0>
  %0 = fabric.fifo %a [max_depth = 1, bypassable = false] : !fabric.bits<0>
  return %0 : !fabric.bits<0>
}

// CHECK-LABEL: @fifo_bits_tag
func.func @fifo_bits_tag(%a: !fabric.bits_tag<8, 2>) -> !fabric.bits_tag<8, 2> {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 4, bypassable = true] : !fabric.bits_tag<8, 2>
  %0 = fabric.fifo %a [max_depth = 4, bypassable = true] : !fabric.bits_tag<8, 2>
  return %0 : !fabric.bits_tag<8, 2>
}

// CHECK-LABEL: @fifo_tag
func.func @fifo_tag(%a: !fabric.tag<3>) -> !fabric.tag<3> {
  // CHECK: fabric.fifo %{{.*}} [max_depth = 2, bypassable = false] : !fabric.tag<3>
  %0 = fabric.fifo %a [max_depth = 2, bypassable = false] : !fabric.tag<3>
  return %0 : !fabric.tag<3>
}
