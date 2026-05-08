// RUN: loom %s | loom | FileCheck %s

// Empty thread definition with no body operands.
// CHECK-LABEL: dataflow.thread private @t_empty()
dataflow.thread private @t_empty() {
  dataflow.thread.yield
}

// Thread definition with two body operands and one trailing iv arg.
// CHECK-LABEL: dataflow.thread private @t_two_args(%{{.*}}: i32, %{{.*}}: f32) iv (%{{.*}}: index)
dataflow.thread private @t_two_args(%a: i32, %b: f32) iv (%i: index) {
  dataflow.thread.yield
}

// Async-token-producing launch carrying mapped operands and a grid
// upper bound.
// CHECK-LABEL: func.func @launch_demo
func.func @launch_demo(%a: i32, %b: f32, %n: index) {
  // CHECK: dataflow.thread.launch @t_two_args(%{{.*}}, %{{.*}}) grid(%{{.*}}) : (i32, f32) -> ()
  dataflow.thread.launch @t_two_args(%a, %b) grid(%n) : (i32, f32) -> ()
  return
}

// Token-producing launch round-trip.
// CHECK-LABEL: func.func @launch_token
func.func @launch_token() -> !dataflow.thread_token {
  // CHECK: %{{.*}} = dataflow.thread.launch @t_empty() : () -> !dataflow.thread_token
  %tok = dataflow.thread.launch @t_empty() : () -> !dataflow.thread_token
  return %tok : !dataflow.thread_token
}
