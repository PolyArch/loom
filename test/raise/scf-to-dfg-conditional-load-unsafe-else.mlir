// RUN: loom-raise-opt --loom-lower-graph-control %s | FileCheck %s

// A conditional-load rewrite that demuxes both lanes is only safe when the
// else yield is already a per-iteration loop-carried value. A standalone
// graph argument is a one-shot token, so the graph-control pass must preserve
// the scf.if envelope instead of lowering the shape to dataflow.demux/mux.

// CHECK-LABEL: dataflow.graph.func private @unsafe_conditional_load_else
// CHECK: %[[IF:.*]] = scf.if
// CHECK-NOT: dataflow.demux
// CHECK-NOT: dataflow.gate
// CHECK: dataflow.graph.return %arg0, %[[IF]] : none, f32
dataflow.graph.func private @unsafe_conditional_load_else(
    %ctrl: none, %cond: i1, %input: memref<?xf32>, %fallback: f32)
    -> (none, f32) {
  %idx = dataflow.constant %ctrl {const_value = 0 : index} : index
  // expected-remark@+1 {{loom-lower-graph-control: scf.if shape not lifted}}
  %next = scf.if %cond -> (f32) {
    %data, %done = dataflow.load %input[%idx] %ctrl : memref<?xf32>
    scf.yield %data : f32
  } else {
    scf.yield %fallback : f32
  }
  dataflow.graph.return %ctrl, %next : none, f32
}
