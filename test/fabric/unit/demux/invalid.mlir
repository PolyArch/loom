// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Fewer than 2 outputs is illegal.
func.func @demux_too_few(%a: !fabric.bits<8>) -> !fabric.bits<8> {
  // expected-error @+1 {{requires at least 2 outputs}}
  %x = "fabric.demux"(%a) : (!fabric.bits<8>) -> !fabric.bits<8>
  return %x : !fabric.bits<8>
}

// -----
// Partial software parameters.
func.func @demux_partial_params(%a: !fabric.bits<8>) -> (!fabric.bits<8>, !fabric.bits<8>) {
  // expected-error @+1 {{software parameters must be all set or all unset}}
  %x, %y = fabric.demux %a {sel = 0 : i32, discard = false} : !fabric.bits<8> -> 2
  return %x, %y : !fabric.bits<8>, !fabric.bits<8>
}

// -----
// discard and disconnect both true is illegal.
func.func @demux_discard_and_disconnect(%a: !fabric.bits<8>) -> (!fabric.bits<8>, !fabric.bits<8>) {
  // expected-error @+1 {{'discard' and 'disconnect' cannot both be true}}
  %x, %y = fabric.demux %a {sel = 0 : i32, discard = true, disconnect = true}
            : !fabric.bits<8> -> 2
  return %x, %y : !fabric.bits<8>, !fabric.bits<8>
}

// -----
// When disconnect is true, sel must be 0.
func.func @demux_disconnect_nonzero_sel(%a: !fabric.bits<8>) -> (!fabric.bits<8>, !fabric.bits<8>) {
  // expected-error @+1 {{when 'disconnect' is true, 'sel' must be 0}}
  %x, %y = fabric.demux %a {sel = 1 : i32, discard = false, disconnect = true}
            : !fabric.bits<8> -> 2
  return %x, %y : !fabric.bits<8>, !fabric.bits<8>
}

// -----
// sel out of [0, N).
func.func @demux_sel_out_of_range(%a: !fabric.bits<8>) -> (!fabric.bits<8>, !fabric.bits<8>) {
  // expected-error @+1 {{'sel' (2) must be in [0, 2)}}
  %x, %y = fabric.demux %a {sel = 2 : i32, discard = false, disconnect = false}
            : !fabric.bits<8> -> 2
  return %x, %y : !fabric.bits<8>, !fabric.bits<8>
}
