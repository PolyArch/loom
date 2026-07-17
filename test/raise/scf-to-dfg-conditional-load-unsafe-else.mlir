// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// A one-shot fallback is explicitly projected into the selected branch domain
// before the conditional load result is merged and published.

// CHECK-LABEL: dataflow.graph private @unsafe_conditional_load_else
// CHECK: %[[FALLBACK:.*]]:2 = dataflow.demux %arg1, %arg2 : (i1, f32) -> (f32, f32)
// CHECK: %[[LOADED:.*]], %[[LOAD_DONE:.*]] = dataflow.load %arg3[{{.*}}]
// CHECK: %[[RESULT:.*]] = dataflow.mux %arg1, %[[FALLBACK]]#0, %[[LOADED]] : (i1, f32, f32) -> f32
// CHECK: dataflow.sync {{.*}}, %[[RESULT]] : (none, f32) -> (none, f32)
// CHECK-NOT: scf.if
dataflow.graph private @unsafe_conditional_load_else(
    %ctrl: none, %cond: i1, %fallback: f32, %input: memref<?xf32>)
    -> (f32)
    attributes {input_segments = array<i32: 2, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %idx = dataflow.constant %ctrl {const_value = 0 : index} : index
  %next = scf.if %cond -> (f32) {
    %data, %done = dataflow.load %input[%idx] %ctrl : memref<?xf32>
    scf.yield %data : f32
  } else {
    scf.yield %fallback : f32
  }
  dataflow.graph.return %ctrl, %next : none, f32
}
