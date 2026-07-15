// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Launch's body operand types must match callee's function inputs.
dataflow.thread private @t_int(%x: i32) ctrl (%c: none) {
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

// -----
// A body-carrying thread must have the `thread_ctrl` slot per spec
// section 5.4.1's `(args_*, thread_ctrl, iv_*)` layout. A thread
// whose entry block has only the function-input args (no ctrl, no
// ivs) is rejected.
// expected-error @+1 {{entry block must have at least 1 arguments (function inputs + 1 thread_ctrl slot)}}
"dataflow.thread"() <{function_type = () -> (), sym_name = "t_no_ctrl",
                     sym_visibility = "private"}> ({
^bb0:
  dataflow.thread.yield
}) : () -> ()

// -----
// thread_ctrl must sit immediately after the function-input args.
// Putting an `index` iv between them is rejected because slot N
// (here index 0, since function_type is empty) must be `none`.
// expected-error @+1 {{entry block argument #0 (thread_ctrl) must have type `none`, got 'index'}}
"dataflow.thread"() <{function_type = () -> (),
                     sym_name = "t_ctrl_wrong_position",
                     sym_visibility = "private"}> ({
^bb0(%i: index, %c: none, %j: index):
  dataflow.thread.yield
}) : () -> ()

// -----
// Grid iv slots must be `index`-typed. A non-index trailing slot
// after the thread_ctrl is rejected.
// expected-error @+1 {{entry block argument #2 (grid iv) must have type `index`, got 'i32'}}
"dataflow.thread"() <{function_type = () -> (),
                     sym_name = "t_iv_wrong_type",
                     sym_visibility = "private"}> ({
^bb0(%c: none, %i: index, %bad: i32):
  dataflow.thread.yield
}) : () -> ()

// -----
// A thread body cannot launch another thread.
dataflow.thread private @nested_thread_leaf() ctrl (%ctrl: none) {
  dataflow.thread.yield
}
dataflow.thread private @nested_thread_parent() ctrl (%ctrl: none) {
  scf.execute_region {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    %token = dataflow.thread.launch @nested_thread_leaf() : () -> !dataflow.thread_token
    scf.yield
  }
  dataflow.thread.yield
}

// -----
// A legacy regional graph body cannot launch a thread either.
dataflow.thread private @regional_graph_thread_leaf() ctrl (%ctrl: none) {
  dataflow.thread.yield
}
func.func @regional_graph_launches_thread() {
  dataflow.graph() -> () {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    %token = dataflow.thread.launch @regional_graph_thread_leaf() : () -> !dataflow.thread_token
    dataflow.yield
  }
  return
}
