// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Launch's body operand types must match callee's function inputs.
dataflow.thread private @t_int(%x: i32) {
  dataflow.thread.yield
}
func.func @launch_type_mismatch(%y: f32) {
  // expected-error @+1 {{body operand #0 type 'f32' does not match callee input type 'i32'}}
  dataflow.thread.launch @t_int(%y) : (f32) -> ()
  return
}

// -----
// Launch must reference a real dataflow.thread symbol.
func.func @launch_unknown_callee() {
  // expected-error @+1 {{'unknown_thread' does not reference a valid 'dataflow.thread' op}}
  dataflow.thread.launch @unknown_thread() : () -> ()
  return
}
