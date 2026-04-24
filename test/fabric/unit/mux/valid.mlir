// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Pure hardware (no software params programmed).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @mux_hw_bits
func.func @mux_hw_bits(%a: !fabric.bits<8>, %b: !fabric.bits<8>) -> !fabric.bits<8> {
  // CHECK: fabric.mux %{{.*}}, %{{.*}} : !fabric.bits<8>
  %0 = fabric.mux %a, %b : !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// CHECK-LABEL: @mux_hw_bits_zero
func.func @mux_hw_bits_zero(%a: !fabric.bits<0>, %b: !fabric.bits<0>, %c: !fabric.bits<0>) -> !fabric.bits<0> {
  // CHECK: fabric.mux %{{.*}}, %{{.*}}, %{{.*}} : !fabric.bits<0>
  %0 = fabric.mux %a, %b, %c : !fabric.bits<0>
  return %0 : !fabric.bits<0>
}

// CHECK-LABEL: @mux_hw_bits_tag
func.func @mux_hw_bits_tag(%a: !fabric.bits_tag<8, 2>, %b: !fabric.bits_tag<8, 2>) -> !fabric.bits_tag<8, 2> {
  // CHECK: fabric.mux %{{.*}}, %{{.*}} : !fabric.bits_tag<8, 2>
  %0 = fabric.mux %a, %b : !fabric.bits_tag<8, 2>
  return %0 : !fabric.bits_tag<8, 2>
}

// CHECK-LABEL: @mux_hw_tag
func.func @mux_hw_tag(%a: !fabric.tag<3>, %b: !fabric.tag<3>) -> !fabric.tag<3> {
  // CHECK: fabric.mux %{{.*}}, %{{.*}} : !fabric.tag<3>
  %0 = fabric.mux %a, %b : !fabric.tag<3>
  return %0 : !fabric.tag<3>
}

// -----------------------------------------------------------------------------
// Programmed: normal pass-through.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @mux_sw_passthrough
func.func @mux_sw_passthrough(%a: !fabric.bits<16>, %b: !fabric.bits<16>, %c: !fabric.bits<16>) -> !fabric.bits<16> {
  // CHECK: fabric.mux %{{.*}}, %{{.*}}, %{{.*}} {sel = 1 : i32, discard = false, disconnect = false} : !fabric.bits<16>
  %0 = fabric.mux %a, %b, %c {sel = 1 : i32, discard = false, disconnect = false} : !fabric.bits<16>
  return %0 : !fabric.bits<16>
}

// -----------------------------------------------------------------------------
// Programmed: discard mode.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @mux_sw_discard
func.func @mux_sw_discard(%a: !fabric.bits<4>, %b: !fabric.bits<4>) -> !fabric.bits<4> {
  // CHECK: fabric.mux %{{.*}}, %{{.*}} {sel = 0 : i32, discard = true, disconnect = false} : !fabric.bits<4>
  %0 = fabric.mux %a, %b {sel = 0 : i32, discard = true, disconnect = false} : !fabric.bits<4>
  return %0 : !fabric.bits<4>
}

// -----------------------------------------------------------------------------
// Programmed: disconnect mode (sel forced to 0).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @mux_sw_disconnect
func.func @mux_sw_disconnect(%a: !fabric.bits<4>, %b: !fabric.bits<4>) -> !fabric.bits<4> {
  // CHECK: fabric.mux %{{.*}}, %{{.*}} {sel = 0 : i32, discard = false, disconnect = true} : !fabric.bits<4>
  %0 = fabric.mux %a, %b {sel = 0 : i32, discard = false, disconnect = true} : !fabric.bits<4>
  return %0 : !fabric.bits<4>
}
