// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Normalized segment sizes must cover every application input.
// expected-error @+1 {{input_segments must contain exactly three nonnegative sizes whose sum (0) matches the function input count (1)}}
dataflow.graph.func private @g_bad_input_segments(%ctrl: none, %x: i32)
    -> (none, i32)
    attributes {input_segments = array<i32: 0, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  dataflow.graph.return %ctrl, %x : none, i32
}

// -----
// A scalar cannot be declared as a memory capability.
// expected-error @+1 {{memory input #0 has non-capability type 'i32'}}
dataflow.graph.func private @g_scalar_memory(%ctrl: none, %x: i32) -> none
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// A memory capability cannot be declared as an application value.
// expected-error @+1 {{value input #0 contains memory capability type 'memref<?xi32>'}}
dataflow.graph.func private @g_memory_value(%ctrl: none,
                                             %memory: memref<?xi32>) -> none
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// Missing segment metadata normalizes to value ports. Pointer syntax must not
// silently redefine the graph ABI.
// expected-error @+1 {{value input #0 contains memory capability type 'memref<?xi32>'}}
dataflow.graph.func private @g_memory_without_classification(
    %ctrl: none, %memory: memref<?xi32>) -> none {
  dataflow.graph.return %ctrl : none
}

// -----
// Capability rejection applies recursively to aggregate value ports.
// expected-error @+1 {{value input #0 contains memory capability type 'tuple<i32, !llvm.ptr>'}}
dataflow.graph.func private @g_nested_memory_value(
    %ctrl: none, %aggregate: tuple<i32, !llvm.ptr>) -> none
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// A scalar cannot be declared as a memory result.
// expected-error @+1 {{memory result #0 has non-capability type 'i32'}}
dataflow.graph.func private @g_scalar_memory_result(%ctrl: none, %x: i32)
    -> (none, i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 1>} {
  dataflow.graph.return values() streams() memories(%x : i32)
      complete(%ctrl : none)
}

// -----
// graph.return value count must match parent results.
dataflow.graph.func private @g_bad_return(%ctrl: none, %x: i32) -> (none, i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  // expected-error @+1 {{values segment count (0) must match parent result segment size (1)}}
  dataflow.graph.return %ctrl : none
}

// -----
// graph.return completion is mandatory.
dataflow.graph.func private @g_empty_complete(%ctrl: none, %x: i32)
    -> (none, i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  // expected-error @+1 {{complete segment must not be empty}}
  dataflow.graph.return values(%x : i32) streams() memories() complete()
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
  scf.execute_region {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    %tok = dataflow.thread.launch @t_empty() : () -> !dataflow.thread_token
    scf.yield
  }
  dataflow.graph.return %ctrl : none
}

// -----
// A graph.func body cannot wait on a caller-side completion token.
dataflow.graph.func private @g_rejects_thread_wait(%ctrl: none) -> none {
  %token = ub.poison : !dataflow.thread_token
  scf.execute_region {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    dataflow.thread.wait %token : !dataflow.thread_token
    scf.yield
  }
  dataflow.graph.return %ctrl : none
}
