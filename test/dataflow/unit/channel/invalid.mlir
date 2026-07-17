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
dataflow.thread private @reject_send_mismatch(
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
