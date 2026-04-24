// RUN: loom %s | loom | FileCheck %s

// Empty graph, no inputs, no results.
// CHECK-LABEL: @graph_empty
func.func @graph_empty() {
  // CHECK: dataflow.graph() -> ()
  dataflow.graph() -> () {
  }
  return
}

// Graph with inputs only, explicit yield with no values.
// CHECK-LABEL: @graph_inputs_only
func.func @graph_inputs_only(%x: i32, %y: f32) {
  // CHECK: dataflow.graph(%{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : f32) -> ()
  dataflow.graph(%a = %x : i32, %b = %y : f32) -> () {
    dataflow.yield
  }
  return
}

// Graph with a simple pipeline: stream -> yield.
// CHECK-LABEL: @graph_stream_pipeline
func.func @graph_stream_pipeline(%lb: i32, %ub: i32, %step: i32) -> (i32, i1) {
  // CHECK: %{{.*}}:2 = dataflow.graph(%{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : i32, %{{.*}} = %{{.*}} : i32) -> (i32, i1)
  %idx, %rwc = dataflow.graph(%l = %lb : i32, %u = %ub : i32, %s = %step : i32) -> (i32, i1) {
    %i, %r = dataflow.stream %l, %u, %s {step_op = "+=", cont_cond = "<"} : i32
    dataflow.yield %i, %r : i32, i1
  }
  return %idx, %rwc : i32, i1
}

// Graph with self-feedback: a carry op whose carry input is its own output.
// CHECK-LABEL: @graph_self_feedback
func.func @graph_self_feedback(%cond: i1, %init: i32) -> i32 {
  // CHECK: %{{.*}} = dataflow.graph(%{{.*}} = %{{.*}} : i1, %{{.*}} = %{{.*}} : i32) -> i32
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32) -> i32 {
    %out = dataflow.carry %c, %i, %out : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

// Graph with a plain backward reference (non-feedback): the op on the first
// line textually uses a value produced on the second line.
// CHECK-LABEL: @graph_backward_ref
func.func @graph_backward_ref(%cond: i1, %init: i32) -> i32 {
  // CHECK: dataflow.graph(%{{.*}} = %{{.*}} : i1, %{{.*}} = %{{.*}} : i32) -> i32
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32) -> i32 {
    // %later is defined further down, but the graph region has no SSA
    // dominance so this forward use is legal.
    %first = dataflow.invariant %c, %later : i32
    %later = dataflow.carry %c, %i, %first : i32
    dataflow.yield %first : i32
  }
  return %r : i32
}

// Nested dataflow.graph: a graph that contains another graph.
// CHECK-LABEL: @graph_nested
func.func @graph_nested(%cond: i1, %init: i32) -> i32 {
  // CHECK: %{{.*}} = dataflow.graph(%{{.*}} = %{{.*}} : i1, %{{.*}} = %{{.*}} : i32) -> i32
  %r = dataflow.graph(%c = %cond : i1, %i = %init : i32) -> i32 {
    // CHECK: %{{.*}} = dataflow.graph(%{{.*}} = %{{.*}} : i1, %{{.*}} = %{{.*}} : i32) -> i32
    %inner = dataflow.graph(%cn = %c : i1, %in = %i : i32) -> i32 {
      %o = dataflow.carry %cn, %in, %o : i32
      dataflow.yield %o : i32
    }
    dataflow.yield %inner : i32
  }
  return %r : i32
}

// Graph over memref + none.
// CHECK-LABEL: @graph_memref
func.func @graph_memref(%mem: memref<16xi32>, %addr: index, %ctrl: none) -> (i32, none) {
  // CHECK: %{{.*}}:2 = dataflow.graph(%{{.*}} = %{{.*}} : memref<16xi32>, %{{.*}} = %{{.*}} : index, %{{.*}} = %{{.*}} : none) -> (i32, none)
  %d, %done = dataflow.graph(%m = %mem : memref<16xi32>, %a = %addr : index, %c = %ctrl : none) -> (i32, none) {
    %dd, %dn = dataflow.load %m[%a] %c : memref<16xi32>
    dataflow.yield %dd, %dn : i32, none
  }
  return %d, %done : i32, none
}
