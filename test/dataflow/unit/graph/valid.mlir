// RUN: loom %s | loom | FileCheck %s
// RUN: loom %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// Graph definition with explicit start/done protocol syntax. The stored
// FunctionType contains only application payloads.
// CHECK-LABEL: dataflow.graph private @g_demo(%{{.*}}: none, %{{.*}}: i32) -> i32
// CHECK-SAME: attributes {input_segments = array<i32: 1, 0, 0>, result_segments = array<i32: 1, 0, 0>}
// GENERIC: function_type = (i32) -> i32
// GENERIC-SAME: input_segments = array<i32: 1, 0, 0>
// GENERIC-SAME: result_segments = array<i32: 1, 0, 0>
dataflow.graph private @g_demo(%ctrl: none, %x: i32) -> i32
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  // CHECK: dataflow.graph.return %{{.*}}, %{{.*}} : none, i32
  dataflow.graph.return %ctrl, %x : none, i32
}

// Multiple completion witnesses use the explicit segmented form.
// CHECK-LABEL: dataflow.graph private @g_segmented
// CHECK: dataflow.graph.return values(%{{.*}} : i32) streams() memories() complete(%{{.*}}, %{{.*}} : none, none)
dataflow.graph private @g_segmented(%ctrl: none, %x: i32) -> i32
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  %done:2 = dataflow.sync %ctrl, %ctrl : (none, none) -> (none, none)
  dataflow.graph.return values(%x : i32) streams() memories()
      complete(%done#0, %done#1 : none, none)
}

// Memory ports are classified by normalized segments rather than by consumers
// rediscovering capability types.
// CHECK-LABEL: dataflow.graph private @g_memory
// CHECK-SAME: attributes {input_segments = array<i32: 1, 0, 1>, result_segments = array<i32: 0, 0, 1>}
dataflow.graph private @g_memory(%ctrl: none, %x: i32,
                                 %memory: memref<?xi32>)
    -> memref<?xi32>
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  dataflow.graph.return values() streams()
      memories(%memory : memref<?xi32>) complete(%ctrl : none)
}

// Asynchronous launch site inside a thread body. Dependencies and payload
// bindings are distinct, and done follows all SSA payload results.
// CHECK-LABEL: dataflow.thread private @t_demo(%{{.*}}: i32) ctrl (%{{.*}}: none)
dataflow.thread private @t_demo(%x: i32) ctrl (%ctrl: none) {
  // CHECK: %{{.*}}, %{{.*}} = dataflow.graph.launch @g_demo deps(%{{.*}}) values(%{{.*}}) stream_inputs() memories() stream_outputs() : (none, i32) -> (i32, none)
  %r, %done = dataflow.graph.launch @g_demo deps(%ctrl) values(%x)
      stream_inputs() memories() stream_outputs()
      : (none, i32) -> (i32, none)
  dataflow.thread.yield
}

// Stream payloads remain graph-local SSA values and bind to thread channel
// endpoints only at launch sites.
// CHECK-LABEL: dataflow.graph private @g_stream
dataflow.graph private @g_stream(%start: none, %input: i32) -> i32
    attributes {input_segments = array<i32: 0, 1, 0>,
                result_segments = array<i32: 0, 1, 0>} {
  dataflow.graph.return values() streams(%input : i32) memories()
      complete(%start : none)
}

// CHECK-LABEL: dataflow.thread private @t_stream
dataflow.thread private @t_stream(
    %input: !dataflow.channel<i32>,
    %output: !dataflow.channel<i32>) ctrl (%ctrl: none) {
  // CHECK: %{{.*}} = dataflow.graph.launch @g_stream deps(%{{.*}}) values() stream_inputs(%{{.*}}) memories() stream_outputs(%{{.*}}) : (none, !dataflow.channel<i32>, !dataflow.channel<i32>) -> none
  %done = dataflow.graph.launch @g_stream deps(%ctrl) values()
      stream_inputs(%input) memories() stream_outputs(%output)
      : (none, !dataflow.channel<i32>, !dataflow.channel<i32>) -> none
  dataflow.thread.yield %done : none
}
