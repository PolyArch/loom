// RUN: loom %s | loom | FileCheck %s

// Empty thread body carrying just the thread_ctrl slot per spec
// section 5.4.1's `(args_*, thread_ctrl, iv_*)` layout.
// CHECK-LABEL: dataflow.thread private @t_empty domain(#dataflow.thread_domain<dense>)() ctrl (%{{.*}}: none)
dataflow.thread private @t_empty domain(#dataflow.thread_domain<dense>)() ctrl (%c: none) {
  dataflow.thread.yield
}

// Thread definition with two body operands, the thread_ctrl slot,
// and one trailing grid iv slot.
// CHECK-LABEL: dataflow.thread private @t_two_args domain(#dataflow.thread_domain<dense>)(%{{.*}}: i32, %{{.*}}: f32) ctrl (%{{.*}}: none) iv (%{{.*}}: index)
dataflow.thread private @t_two_args domain(#dataflow.thread_domain<dense>)(%a: i32, %b: f32) ctrl (%c: none) iv (%i: index) {
  dataflow.thread.yield
}

// Every launch produces one completion token, including a launch carrying
// mapped operands and a grid upper bound.
// CHECK-LABEL: func.func @launch_demo
func.func @launch_demo(%a: i32, %b: f32, %n: index) {
  // CHECK: %{{.*}} = dataflow.thread.launch @t_two_args(%{{.*}}, %{{.*}}) grid(%{{.*}}) : (i32, f32) -> !dataflow.thread_token
  %completion = dataflow.thread.launch @t_two_args(%a, %b) grid(%n) : (i32, f32) -> !dataflow.thread_token
  return
}

// Launch dependencies and waits both express unordered all-of completion.
// CHECK-LABEL: func.func @wait_for_launches
func.func @wait_for_launches() {
  // CHECK: %{{.*}} = dataflow.thread.launch @t_empty() : () -> !dataflow.thread_token
  %first = dataflow.thread.launch @t_empty() : () -> !dataflow.thread_token
  // CHECK: %{{.*}} = dataflow.thread.launch @t_empty() wait(%{{.*}}) : () -> !dataflow.thread_token
  %second = dataflow.thread.launch @t_empty() wait(%first) : () -> !dataflow.thread_token
  // CHECK: dataflow.thread.wait %{{.*}}, %{{.*}} : !dataflow.thread_token, !dataflow.thread_token
  dataflow.thread.wait %first, %second : !dataflow.thread_token, !dataflow.thread_token
  return
}

// Completion frontiers carry zero or more unordered none-typed values.
// CHECK-LABEL: dataflow.thread private @t_frontier domain(#dataflow.thread_domain<dense>)() ctrl (%{{.*}}: none)
dataflow.thread private @t_frontier domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
  // CHECK: dataflow.thread.yield %{{.*}} : none
  dataflow.thread.yield %ctrl : none
}

// A dynamic-work definition designates one ordinary input as its root payload
// and has no coordinate suffix or launch extents.
// CHECK-LABEL: dataflow.thread private @t_dynamic domain(#dataflow.thread_domain<dynamic_work, work_item_arg = 0>)
dataflow.thread private @t_dynamic domain(#dataflow.thread_domain<dynamic_work, work_item_arg = 0>)(%work: i32) ctrl (%ctrl: none) {
  dataflow.thread.yield
}

// CHECK-LABEL: func.func @launch_dynamic
func.func @launch_dynamic(%root: i32) {
  // CHECK: dataflow.thread.launch @t_dynamic(%{{.*}}) : (i32) -> !dataflow.thread_token
  %completion = dataflow.thread.launch @t_dynamic(%root) : (i32) -> !dataflow.thread_token
  return
}
