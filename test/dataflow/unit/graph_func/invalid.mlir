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

// -----
// graph.func body whitelist: a direct func.call into a host symbol is
// rejected. Calls into ScalarCore must go through dataflow.thread.launch
// or dataflow.graph.launch, never through func.call.
func.func private @host_helper(%x: i32) -> i32
dataflow.graph.func private @g_rejects_func_call(%ctrl: none, %x: i32)
    -> (none, i32) {
  // expected-error @+1 {{is not allowed inside a dataflow.graph.func body}}
  %r = func.call @host_helper(%x) : (i32) -> i32
  dataflow.graph.return %ctrl, %r : none, i32
}

// -----
// graph.func body whitelist: nested function-symbol definitions do not
// belong inside a graph.func body. The body is leaf SpatialCore
// compute, not a place to anchor further symbol definitions.
dataflow.graph.func private @g_rejects_nested_func(%ctrl: none) -> none {
  // expected-error @+1 {{is not allowed inside a dataflow.graph.func body}}
  func.func private @inner_def(%y: i32) -> i32 {
    func.return %y : i32
  }
  dataflow.graph.return %ctrl : none
}

// -----
// graph.func body whitelist: a tensor-dialect op (not on the SCF-to-DFG
// residual surface) is rejected. tensor.* is loaded by the loom driver
// because it ships with mlir, so this exercises the dialect-membership
// gate without requiring a synthetic test op.
dataflow.graph.func private @g_rejects_tensor_op(%ctrl: none) -> (none, tensor<1xi32>) {
  // expected-error @+1 {{is not allowed inside a dataflow.graph.func body}}
  %t = tensor.empty() : tensor<1xi32>
  dataflow.graph.return %ctrl, %t : none, tensor<1xi32>
}

// -----
// graph.func body whitelist: thread launches belong in the host-level
// launch layer, not inside leaf SpatialCore graph bodies.
dataflow.thread private @t_empty() ctrl (%thread_ctrl: none) {
  dataflow.thread.yield
}
dataflow.graph.func private @g_rejects_thread_launch(%ctrl: none) -> none {
  // expected-error @+1 {{is not allowed inside a dataflow.graph.func body}}
  %tok = dataflow.thread.launch @t_empty() : () -> !dataflow.thread_token
  dataflow.graph.return %ctrl : none
}
