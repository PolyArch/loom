// RUN: loom %s | loom | FileCheck %s

// Graph definition with leading none ctrl_in and none done_out.
// CHECK-LABEL: dataflow.graph.func private @g_demo(%{{.*}}: none, %{{.*}}: i32) -> (none, i32)
dataflow.graph.func private @g_demo(%ctrl: none, %x: i32) -> (none, i32) {
  // CHECK: dataflow.graph.return %{{.*}}, %{{.*}} : none, i32
  dataflow.graph.return %ctrl, %x : none, i32
}

// Synchronous launch site inside a thread body. The launch's first
// operand and result are the per-launch ctrl_in / done_out ports.
dataflow.thread private @t_demo(%x: i32) {
  // CHECK: %{{.*}} = ub.poison : none
  %ctrl = ub.poison : none
  // CHECK: %{{.*}}, %{{.*}} = dataflow.graph.launch @g_demo(%{{.*}}, %{{.*}}) : (none, i32) -> (none, i32)
  %done, %r = dataflow.graph.launch @g_demo(%ctrl, %x) : (none, i32) -> (none, i32)
  dataflow.thread.yield
}
