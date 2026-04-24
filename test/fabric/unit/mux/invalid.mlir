// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Fewer than 2 inputs is illegal.
func.func @mux_too_few(%a: !fabric.bits<8>) -> !fabric.bits<8> {
  // expected-error @+1 {{requires at least 2 inputs}}
  %0 = "fabric.mux"(%a) : (!fabric.bits<8>) -> !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----
// Partial software parameters (violates all-or-nothing rule).
func.func @mux_partial_params(%a: !fabric.bits<8>, %b: !fabric.bits<8>) -> !fabric.bits<8> {
  // expected-error @+1 {{software parameters must be all set or all unset}}
  %0 = fabric.mux %a, %b {sel = 0 : i32, discard = false} : !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----
// discard and disconnect both true is illegal.
func.func @mux_discard_and_disconnect(%a: !fabric.bits<8>, %b: !fabric.bits<8>) -> !fabric.bits<8> {
  // expected-error @+1 {{'discard' and 'disconnect' cannot both be true}}
  %0 = fabric.mux %a, %b {sel = 0 : i32, discard = true, disconnect = true} : !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----
// When disconnect is true, sel must be 0.
func.func @mux_disconnect_nonzero_sel(%a: !fabric.bits<8>, %b: !fabric.bits<8>) -> !fabric.bits<8> {
  // expected-error @+1 {{when 'disconnect' is true, 'sel' must be 0}}
  %0 = fabric.mux %a, %b {sel = 1 : i32, discard = false, disconnect = true} : !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----
// sel out of [0, N).
func.func @mux_sel_out_of_range(%a: !fabric.bits<8>, %b: !fabric.bits<8>) -> !fabric.bits<8> {
  // expected-error @+1 {{'sel' (2) must be in [0, 2)}}
  %0 = fabric.mux %a, %b {sel = 2 : i32, discard = false, disconnect = false} : !fabric.bits<8>
  return %0 : !fabric.bits<8>
}

// -----
// bits_tag requires width > 0.
// expected-error @+1 {{fabric.bits_tag requires width > 0}}
func.func private @bad_bits_tag_zero_width(!fabric.bits_tag<0, 2>)

// -----
// tag requires tagWidth > 0.
// expected-error @+1 {{fabric.tag requires tagWidth > 0}}
func.func private @bad_tag_zero_width(!fabric.tag<0>)
