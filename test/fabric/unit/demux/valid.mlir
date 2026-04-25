// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Pure hardware (no software params programmed).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @demux_hw_bits
func.func @demux_hw_bits(%a: !fabric.bits<8>) -> (!fabric.bits<8>, !fabric.bits<8>) {
  // CHECK: fabric.demux %{{.*}} : !fabric.bits<8> -> 2
  %x, %y = fabric.demux %a : !fabric.bits<8> -> 2
  return %x, %y : !fabric.bits<8>, !fabric.bits<8>
}

// CHECK-LABEL: @demux_hw_bits_zero
func.func @demux_hw_bits_zero(%a: !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
  // CHECK: fabric.demux %{{.*}} : !fabric.bits<0> -> 3
  %x, %y, %z = fabric.demux %a : !fabric.bits<0> -> 3
  return %x, %y, %z : !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>
}

// CHECK-LABEL: @demux_hw_bits_tag
func.func @demux_hw_bits_tag(%a: !fabric.bits_tag<8, 2>) -> (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>) {
  // CHECK: fabric.demux %{{.*}} : !fabric.bits_tag<8, 2> -> 2
  %x, %y = fabric.demux %a : !fabric.bits_tag<8, 2> -> 2
  return %x, %y : !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>
}

// CHECK-LABEL: @demux_hw_tag
func.func @demux_hw_tag(%a: !fabric.tag<3>) -> (!fabric.tag<3>, !fabric.tag<3>) {
  // CHECK: fabric.demux %{{.*}} : !fabric.tag<3> -> 2
  %x, %y = fabric.demux %a : !fabric.tag<3> -> 2
  return %x, %y : !fabric.tag<3>, !fabric.tag<3>
}

// -----------------------------------------------------------------------------
// Programmed: route to selected output.
// -----------------------------------------------------------------------------

// CHECK-LABEL: @demux_sw_route
func.func @demux_sw_route(%a: !fabric.bits<16>) -> (!fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>) {
  // CHECK: fabric.demux %{{.*}} {sel = 1 : i32, discard = false, disconnect = false} : !fabric.bits<16> -> 3
  %x, %y, %z = fabric.demux %a {sel = 1 : i32, discard = false, disconnect = false}
                : !fabric.bits<16> -> 3
  return %x, %y, %z : !fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>
}

// -----------------------------------------------------------------------------
// Programmed: discard mode (input drained, no output token).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @demux_sw_discard
func.func @demux_sw_discard(%a: !fabric.bits<4>) -> (!fabric.bits<4>, !fabric.bits<4>) {
  // CHECK: fabric.demux %{{.*}} {sel = 0 : i32, discard = true, disconnect = false} : !fabric.bits<4> -> 2
  %x, %y = fabric.demux %a {sel = 0 : i32, discard = true, disconnect = false}
            : !fabric.bits<4> -> 2
  return %x, %y : !fabric.bits<4>, !fabric.bits<4>
}

// -----------------------------------------------------------------------------
// Programmed: disconnect mode (sel forced to 0).
// -----------------------------------------------------------------------------

// CHECK-LABEL: @demux_sw_disconnect
func.func @demux_sw_disconnect(%a: !fabric.bits<4>) -> (!fabric.bits<4>, !fabric.bits<4>) {
  // CHECK: fabric.demux %{{.*}} {sel = 0 : i32, discard = false, disconnect = true} : !fabric.bits<4> -> 2
  %x, %y = fabric.demux %a {sel = 0 : i32, discard = false, disconnect = true}
            : !fabric.bits<4> -> 2
  return %x, %y : !fabric.bits<4>, !fabric.bits<4>
}
