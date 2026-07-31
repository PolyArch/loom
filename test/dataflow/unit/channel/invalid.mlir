// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// expected-error @+2 {{channel element type must not be another dataflow channel}}
func.func @reject_nested_channel(
    %channel: !dataflow.channel<!dataflow.channel<i32>>) {
  return
}

// -----
// expected-error @+2 {{channel element type must not be !dataflow.thread_token}}
func.func @reject_thread_token_payload(
    %channel: !dataflow.channel<!dataflow.thread_token>) {
  return
}

// -----
// expected-error @+2 {{channel element type must not contain !dataflow.channel or !dataflow.thread_token}}
func.func @reject_nested_channel_tuple(
    %channel: !dataflow.channel<tuple<!dataflow.channel<i32>>>) {
  return
}

// -----
// expected-error @+2 {{channel element type must not contain a memory capability}}
func.func @reject_memref_payload(
    %channel: !dataflow.channel<memref<4xi32>>) {
  return
}

// -----
// expected-error @+2 {{channel element type must not contain a nested LLVM pointer value}}
func.func @reject_nested_pointer_payload(
    %channel: !dataflow.channel<tuple<i32, !llvm.ptr>>) {
  return
}

// -----
dataflow.thread private @reject_send_mismatch domain(#dataflow.thread_domain<dense>)(
    %channel: !dataflow.channel<i32>, %message: i64) ctrl (%ctrl: none) {
  // expected-error @+1 {{failed to verify that 'message' type matches channel element type}}
  "dataflow.channel.send"(%channel, %message)
      : (!dataflow.channel<i32>, i64) -> ()
  dataflow.thread.yield
}

// -----
func.func @reject_receive_outside_thread(
    %channel: !dataflow.channel<i32>) {
  // expected-error @+1 {{must appear inside a dataflow.thread body}}
  %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
  return
}

// -----
dataflow.graph private @reject_receive_in_graph(%start: none) -> () {
  %channel = ub.poison : !dataflow.channel<i32>
  // expected-error @+1 {{must not appear inside a dataflow.graph definition}}
  %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
  dataflow.graph.return %start : none
}

// -----
// expected-error @+1 {{function_type input #0 must not be a dataflow channel type}}
dataflow.graph private @reject_graph_channel_input(
    %start: none, %channel: !dataflow.channel<i32>) -> () {
  dataflow.graph.return %start : none
}

// -----
// expected-error @+1 {{function_type result #0 must not contain !dataflow.channel or !dataflow.thread_token}}
dataflow.graph private @reject_graph_nested_channel_result(%start: none)
    -> tuple<!dataflow.channel<i32>> {
  %payload = ub.poison : tuple<!dataflow.channel<i32>>
  dataflow.graph.return %start, %payload
      : none, tuple<!dataflow.channel<i32>>
}

// -----
// expected-error @+1 {{function_type input #0 must not contain !dataflow.channel or !dataflow.thread_token}}
dataflow.graph private @reject_graph_nested_thread_token(
    %start: none, %payload: tuple<!dataflow.thread_token>) -> () {
  dataflow.graph.return %start : none
}

// -----
dataflow.graph private @stream_source(%start: none, %input: i32) -> ()
    attributes {input_segments = array<i32: 0, 1, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %start : none
}

dataflow.thread private @reject_missing_source_map domain(#dataflow.thread_domain<dense>)(
    %input: !dataflow.channel<i32>) ctrl (%ctrl: none) {
  // expected-error @+2 {{expected 'source_map' after stream input binding}}
  %done = dataflow.graph.launch @stream_source deps(%ctrl) values()
      stream_inputs(%input) memories() stream_outputs()
      : (none, !dataflow.channel<i32>) -> none
  dataflow.thread.yield
}

// -----
dataflow.graph private @ranked_stream_source(%start: none, %input: i32) -> ()
    attributes {input_segments = array<i32: 0, 1, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %start : none
}

dataflow.thread private @reject_source_map_consumer_rank domain(#dataflow.thread_domain<dense>)(
    %input: !dataflow.channel<i32>) ctrl (%ctrl: none) iv (%iv: index) {
  // expected-error @+1 {{stream input source_map #0 has 0 dimensions but consumer thread domain has rank 1}}
  %done = dataflow.graph.launch @ranked_stream_source deps(%ctrl) values()
      stream_inputs(%input source_map affine_map<() -> ()>)
      memories() stream_outputs()
      : (none, !dataflow.channel<i32>) -> none
  dataflow.thread.yield
}

// -----
dataflow.graph private @symbolic_stream_source(%start: none, %input: i32) -> ()
    attributes {input_segments = array<i32: 0, 1, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  dataflow.graph.return %start : none
}

dataflow.thread private @reject_symbolic_source_map domain(#dataflow.thread_domain<dense>)(
    %input: !dataflow.channel<i32>) ctrl (%ctrl: none) {
  // expected-error @+1 {{stream input source_map #0 must not contain symbols}}
  %done = dataflow.graph.launch @symbolic_stream_source deps(%ctrl) values()
      stream_inputs(%input source_map affine_map<()[s0] -> (s0)>)
      memories() stream_outputs()
      : (none, !dataflow.channel<i32>) -> none
  dataflow.thread.yield
}
