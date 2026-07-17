// RUN: loom %s | loom | FileCheck %s
// RUN: loom %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// Graph definition with explicit start/done protocol syntax. The stored
// FunctionType contains only application payloads.
// CHECK-LABEL: dataflow.graph.func private @g_demo(%{{.*}}: none, %{{.*}}: i32) -> (none, i32)
// CHECK-SAME: attributes {input_segments = array<i32: 1, 0, 0>, result_segments = array<i32: 1, 0, 0>}
// GENERIC: function_type = (i32) -> i32
// GENERIC-SAME: input_segments = array<i32: 1, 0, 0>
// GENERIC-SAME: result_segments = array<i32: 1, 0, 0>
dataflow.graph.func private @g_demo(%ctrl: none, %x: i32) -> (none, i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  // CHECK: dataflow.graph.return %{{.*}}, %{{.*}} : none, i32
  dataflow.graph.return %ctrl, %x : none, i32
}

// Multiple completion witnesses use the explicit segmented form.
// CHECK-LABEL: dataflow.graph.func private @g_segmented
// CHECK: dataflow.graph.return values(%{{.*}} : i32) streams() memories() complete(%{{.*}}, %{{.*}} : none, none)
dataflow.graph.func private @g_segmented(%ctrl: none, %x: i32)
    -> (none, i32)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  %done:2 = dataflow.sync %ctrl, %ctrl : (none, none) -> (none, none)
  dataflow.graph.return values(%x : i32) streams() memories()
      complete(%done#0, %done#1 : none, none)
}

// Memory ports are classified by normalized segments rather than by consumers
// rediscovering capability types.
// CHECK-LABEL: dataflow.graph.func private @g_memory
// CHECK-SAME: attributes {input_segments = array<i32: 1, 0, 1>, result_segments = array<i32: 0, 0, 1>}
dataflow.graph.func private @g_memory(%ctrl: none, %x: i32,
                                      %memory: memref<?xi32>)
    -> (none, memref<?xi32>)
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  dataflow.graph.return values() streams()
      memories(%memory : memref<?xi32>) complete(%ctrl : none)
}

// Asynchronous launch site inside a thread body. The launch's first
// operand is the enclosing thread's `thread_ctrl` block argument
// (per spec section 5.4.1); its first result is the per-launch
// `done_out`.
// CHECK-LABEL: dataflow.thread private @t_demo(%{{.*}}: i32) ctrl (%{{.*}}: none)
dataflow.thread private @t_demo(%x: i32) ctrl (%ctrl: none) {
  // CHECK: %{{.*}}, %{{.*}} = dataflow.graph.launch @g_demo(%{{.*}}, %{{.*}}) : (none, i32) -> (none, i32)
  %done, %r = dataflow.graph.launch @g_demo(%ctrl, %x) : (none, i32) -> (none, i32)
  dataflow.thread.yield
}
