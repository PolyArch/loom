// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// graph.func's function_type must lead with `none` ctrl_in.
// expected-error @+1 {{function_type inputs must lead with a `none` ctrl_in slot}}
dataflow.graph.func private @g_no_ctrl(%x: i32) -> (none, i32) {
  %z = ub.poison : none
  dataflow.graph.return %z, %x : none, i32
}

// -----
// graph.func's function_type must lead with `none` done_out.
// expected-error @+1 {{function_type results must lead with a `none` done_out slot}}
dataflow.graph.func private @g_no_done(%ctrl: none, %x: i32) -> i32 {
  dataflow.graph.return %x : i32
}

// -----
// graph.return value count must match parent results.
dataflow.graph.func private @g_bad_return(%ctrl: none, %x: i32) -> (none, i32) {
  // expected-error @+1 {{return value count (1) must match parent dataflow.graph.func result count (2)}}
  dataflow.graph.return %ctrl : none
}

// -----
// graph.launch must appear inside a dataflow.thread body. (Outside
// any thread there is no thread_ctrl slot, so we materialise an
// `ub.poison : none` as a placeholder ctrl_in just to exercise the
// "must be inside a thread" check.)
dataflow.graph.func private @g_target(%ctrl: none) -> none {
  dataflow.graph.return %ctrl : none
}
func.func @launch_outside_thread() {
  %ctrl = ub.poison : none
  // expected-error @+1 {{must appear inside a dataflow.thread body}}
  %d = dataflow.graph.launch @g_target(%ctrl) : (none) -> none
  return
}
