// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// Graph definitions require explicit private visibility.
// expected-error @+1 {{requires explicit 'private' visibility}}
dataflow.graph @g_missing_visibility(%ctrl: none) -> () {
  dataflow.graph.return %ctrl : none
}

// -----
// Normalized segment sizes must cover every application input.
// expected-error @+1 {{input_segments must contain exactly three nonnegative sizes whose sum (0) matches the function input count (1)}}
dataflow.graph private @g_bad_input_segments(%ctrl: none, %x: i32)
    -> (i32)
    attributes {input_segments = array<i32: 0, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  dataflow.graph.return %ctrl, %x : none, i32
}

// -----
// Input segment vectors have exactly value/stream/memory entries.
// expected-error @+1 {{input_segments must contain exactly three nonnegative sizes whose sum (1) matches the function input count (1)}}
dataflow.graph private @g_short_input_segments(%ctrl: none, %x: i32) -> ()
    attributes {input_segments = array<i32: 1, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// Result segment vectors have exactly value/stream/memory entries.
// expected-error @+1 {{result_segments must contain exactly three nonnegative sizes whose sum (1) matches the function result count (1)}}
dataflow.graph private @g_long_result_segments(%ctrl: none, %x: i32) -> i32
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0, 0>} {
  dataflow.graph.return %ctrl, %x : none, i32
}

// -----
// A scalar cannot be declared as a memory capability.
// expected-error @+1 {{memory input #0 has non-capability type 'i32'}}
dataflow.graph private @g_scalar_memory(%ctrl: none, %x: i32) -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// LLVM pointers remain legal in threads but are not canonical graph memory
// ports.
// expected-error @+1 {{memory input #0 must be a memref capability, but got '!llvm.ptr'}}
dataflow.graph private @g_pointer_memory(%ctrl: none, %memory: !llvm.ptr) -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// Graph memory results likewise expose typed memrefs, never raw pointers.
// expected-error @+1 {{memory result #0 must be a memref capability, but got '!llvm.ptr'}}
dataflow.graph private @g_pointer_memory_result(%ctrl: none,
                                                 %memory: memref<?xi32>)
    -> !llvm.ptr
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  %raw = builtin.unrealized_conversion_cast %memory
      : memref<?xi32> to !llvm.ptr
  dataflow.graph.return values() streams() memories(%raw : !llvm.ptr)
      complete(%ctrl : none)
}

// -----
// A memory capability cannot be declared as an application value.
// expected-error @+1 {{value input #0 contains memory capability type 'memref<?xi32>'}}
dataflow.graph private @g_memory_value(%ctrl: none,
                                             %memory: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// Missing segment metadata normalizes to value ports. Pointer syntax must not
// silently redefine the graph ABI.
// expected-error @+1 {{value input #0 contains memory capability type 'memref<?xi32>'}}
dataflow.graph private @g_memory_without_classification(
    %ctrl: none, %memory: memref<?xi32>) -> () {
  dataflow.graph.return %ctrl : none
}

// -----
// Pointer values are scalar transfer payloads rather than aggregate aliases.
// expected-error @+1 {{value input #0 contains a nested LLVM pointer value in type 'tuple<i32, !llvm.ptr>'}}
dataflow.graph private @g_nested_memory_value(
    %ctrl: none, %aggregate: tuple<i32, !llvm.ptr>) -> ()
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}

// -----
// A scalar cannot be declared as a memory result.
// expected-error @+1 {{memory result #0 has non-capability type 'i32'}}
dataflow.graph private @g_scalar_memory_result(%ctrl: none, %x: i32)
    -> (i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 1>} {
  dataflow.graph.return values() streams() memories(%x : i32)
      complete(%ctrl : none)
}

// -----
// graph.return value count must match parent results.
dataflow.graph private @g_bad_return(%ctrl: none, %x: i32) -> (i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  // expected-error @+1 {{values segment count (0) must match parent result segment size (1)}}
  dataflow.graph.return %ctrl : none
}

// -----
// graph.return completion is mandatory.
dataflow.graph private @g_empty_complete(%ctrl: none, %x: i32)
    -> (i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  // expected-error @+1 {{complete segment must not be empty}}
  dataflow.graph.return values(%x : i32) streams() memories() complete()
}

// -----
// graph.launch is owned by exactly one dataflow.thread definition.
// (Outside any thread there is no thread_ctrl slot, so we materialise an
// `ub.poison : none` as a placeholder dependency just to exercise the
// ownership check.)
dataflow.graph private @g_target(%ctrl: none) -> () {
  dataflow.graph.return %ctrl : none
}
func.func @launch_outside_thread() {
  %ctrl = ub.poison : none
  // expected-error @+1 {{must be transitively contained by exactly one dataflow.thread definition}}
  %d = dataflow.graph.launch @g_target deps(%ctrl) values()
      stream_inputs() memories() stream_outputs() : (none) -> none
  return
}

// -----
// A pointer value cannot be silently reinterpreted as a memory capability.
dataflow.graph private @g_linear_import(%ctrl: none, %memory: memref<?xf32>)
    -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %ctrl : none
}
dataflow.thread private @t_pointer_memory_import
    domain(#dataflow.thread_domain<dense>)(%pointer: !llvm.ptr)
    ctrl (%ctrl: none) {
  // expected-error @+1 {{memory input #0 type '!llvm.ptr' does not match callee payload type 'memref<?xf32>'}}
  %done = dataflow.graph.launch @g_linear_import deps(%ctrl) values()
      stream_inputs() memories(%pointer) stream_outputs()
      : (none, !llvm.ptr) -> none
  dataflow.thread.yield
}

// -----
// Two enclosing thread definitions leave graph launch ownership
// ambiguous, so the launch is rejected rather than silently attributed
// to the outer thread.
dataflow.thread private @t_launch_nested_outer domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
  builtin.module {
    dataflow.graph private @g_nested(%start: none) -> ()
        attributes {input_segments = array<i32: 0, 0, 0>,
                    result_segments = array<i32: 0, 0, 0>} {
      dataflow.graph.return %start : none
    }
    dataflow.thread private @t_launch_nested_inner domain(#dataflow.thread_domain<dense>)() ctrl (%inner_ctrl: none) {
      // expected-error @+1 {{must be transitively contained by exactly one dataflow.thread definition}}
      %done = dataflow.graph.launch @g_nested deps(%inner_ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      dataflow.thread.yield %done : none
    }
  }
  dataflow.thread.yield
}

// -----
// Thread launches belong in the host-level
// launch layer, not inside leaf SpatialCore graph bodies.
dataflow.thread private @t_empty domain(#dataflow.thread_domain<dense>)() ctrl (%thread_ctrl: none) {
  dataflow.thread.yield
}
dataflow.graph private @g_rejects_thread_launch(%ctrl: none) -> () {
  scf.execute_region {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    %tok = dataflow.thread.launch @t_empty() : () -> !dataflow.thread_token
    scf.yield
  }
  dataflow.graph.return %ctrl : none
}

// -----
// A graph body cannot wait on a caller-side completion token.
dataflow.graph private @g_rejects_thread_wait(%ctrl: none) -> () {
  %token = ub.poison : !dataflow.thread_token
  scf.execute_region {
    // expected-error @+1 {{must appear outside any dataflow.thread or dataflow.graph definition}}
    dataflow.thread.wait %token : !dataflow.thread_token
    scf.yield
  }
  dataflow.graph.return %ctrl : none
}

// -----
// A graph wait consumes at least one completion event.
dataflow.thread private @t_wait_empty domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
  // expected-error @+1 {{expected 1 or more operands, but found 0}}
  "dataflow.graph.wait"() : () -> ()
  dataflow.thread.yield
}

// -----
// A graph wait at host scope is not contained by any thread definition.
func.func @wait_at_host_scope() {
  %done = ub.poison : none
  // expected-error @+1 {{must be transitively contained by exactly one dataflow.thread definition}}
  dataflow.graph.wait %done : none
  return
}

// -----
// A graph wait cannot appear inside a graph body.
dataflow.graph private @g_rejects_graph_wait(%ctrl: none) -> () {
  scf.execute_region {
    // expected-error @+1 {{must not appear inside a dataflow.graph definition}}
    dataflow.graph.wait %ctrl : none
    scf.yield
  }
  dataflow.graph.return %ctrl : none
}

// -----
// Each thread definition here has the module parent its own trait
// requires, so the wait is reached by the placement check and rejected
// for being transitively contained by two thread definitions.
dataflow.thread private @t_wait_nested_outer domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
  builtin.module {
    dataflow.thread private @t_wait_nested_inner domain(#dataflow.thread_domain<dense>)() ctrl (%inner_ctrl: none) {
      // expected-error @+1 {{must be transitively contained by exactly one dataflow.thread definition}}
      dataflow.graph.wait %inner_ctrl : none
      dataflow.thread.yield
    }
  }
  dataflow.thread.yield
}

// -----
// A graph wait frontier event is a none-typed completion event.
dataflow.thread private @t_wait_wrong_operand_type domain(#dataflow.thread_domain<dense>)(%x: i32) ctrl (%ctrl: none) {
  // expected-error @+1 {{operand #0 must be variadic of none type, but got 'i32'}}
  dataflow.graph.wait %x : i32
  dataflow.thread.yield
}
