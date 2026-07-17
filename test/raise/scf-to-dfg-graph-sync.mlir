// RUN: loom-raise-opt --loom-lower-graph-sync --verify-diagnostics %s | FileCheck %s --implicit-check-not=loom.conditional_store_

// Positive case: a graph.func body with two top-level dataflow.load
// ops produces a single dataflow.sync that consumes both `done`
// tokens, and the graph.return's leading done_out slot is rewritten
// to the sync's first output.

// CHECK-LABEL: dataflow.graph.func private @g_sync_two_loads
// CHECK: %[[D1:.*]], %[[O1:.*]] = dataflow.load %{{.*}}[%c0]
// CHECK: %[[D2:.*]], %[[O2:.*]] = dataflow.load %{{.*}}[%c0]
// CHECK: %[[SYNC:.*]]:2 = dataflow.sync %[[O1]], %[[O2]] : (none, none) -> (none, none)
// CHECK: dataflow.graph.return %[[SYNC]]#0
dataflow.graph.func private @g_sync_two_loads(%arg0: none, %arg1: i64,
                                              %arg2: i64, %arg3: i64,
                                              %arg4: memref<?xf32>,
                                              %arg5: memref<?xf32>)
    -> (none, f32) {
  %c0 = arith.constant 0 : index
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %d1, %o1 = dataflow.load %arg4[%c0] %arg0 : memref<?xf32>
  %d2, %o2 = dataflow.load %arg5[%c0] %arg0 : memref<?xf32>
  %sum = arith.addf %d1, %d2 : f32
  dataflow.graph.return %arg0, %sum : none, f32
}

// Negative-bail: a graph.func body with no dataflow.load and no
// dataflow.store ops emits no sync. The graph.return is left alone.

// CHECK-LABEL: dataflow.graph.func private @g_no_memory_ops
// CHECK-NOT: dataflow.sync
// CHECK: dataflow.graph.return %arg0
dataflow.graph.func private @g_no_memory_ops(%arg0: none, %arg1: i64,
                                             %arg2: i64, %arg3: i64,
                                             %arg4: f32) -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %0 = dataflow.invariant %rwc, %arg4 : f32
  dataflow.graph.return %arg0, %0 : none, f32
}

// A predicated store whose ctrl is routed through dataflow.demux needs one
// completion token per selected or skipped lane. Graph sync must collect the
// merged conditional completion token instead of raw store_done.

// CHECK-LABEL: dataflow.graph.func private @g_predicated_store_done
// CHECK: %[[CTRL_LANES:.*]]:2 = dataflow.demux %arg1, %arg0 : (i1, none) -> (none, none)
// CHECK: %[[STORE_DONE:.*]] = dataflow.store {{.*}} %[[CTRL_LANES]]#1
// CHECK: %[[MERGED_DONE:.*]] = dataflow.mux %arg1, %[[CTRL_LANES]]#0, %[[STORE_DONE]] : (i1, none, none) -> none
// CHECK: dataflow.sync %[[MERGED_DONE]] : (none) -> none
// CHECK: dataflow.graph.return
dataflow.graph.func private @g_predicated_store_done(%arg0: none, %arg1: i1,
                                                     %arg2: memref<?xi32>,
                                                     %arg3: index,
                                                     %arg4: i32) -> none {
  %ctrl_lanes:2 = dataflow.demux %arg1, %arg0 : (i1, none) -> (none, none)
  %store_done = dataflow.store %arg2[%arg3] %arg4 %ctrl_lanes#1
      : memref<?xi32>
  dataflow.graph.return %arg0 : none
}

// An existing canonical completion mux is recognized from topology even when
// it appears before the store in the graph block. Graph sync must reuse it.

// CHECK-LABEL: dataflow.graph.func private @g_reordered_predicated_store_done
// CHECK: %[[REORDERED_CTRL:.*]]:2 = dataflow.demux %arg1, %arg0 : (i1, none) -> (none, none)
// CHECK: %[[REORDERED_DONE:.*]] = dataflow.mux %arg1, %[[REORDERED_CTRL]]#0, %[[REORDERED_STORE:.*]] : (i1, none, none) -> none
// CHECK: %[[REORDERED_STORE]] = dataflow.store {{.*}} %[[REORDERED_CTRL]]#1
// CHECK-NOT: dataflow.mux
// CHECK: dataflow.sync %[[REORDERED_DONE]] : (none) -> none
dataflow.graph.func private @g_reordered_predicated_store_done(
    %arg0: none, %arg1: i1, %arg2: memref<?xi32>, %arg3: index,
    %arg4: i32) -> none {
  %ctrl_lanes:2 = dataflow.demux %arg1, %arg0
      : (i1, none) -> (none, none)
  %merged_done = dataflow.mux %arg1, %ctrl_lanes#0, %store_done
      : (i1, none, none) -> none
  %store_done = dataflow.store %arg2[%arg3] %arg4 %ctrl_lanes#1
      : memref<?xi32>
  dataflow.graph.return %arg0 : none
}

// A nested mux cannot define the graph entry completion event. It is ignored,
// and graph sync materializes a top-level canonical completion mux.

// CHECK-LABEL: dataflow.graph.func private @g_nested_predicated_store_done
// CHECK: %[[NESTED_CTRL:.*]]:2 = dataflow.demux %arg1, %arg0 : (i1, none) -> (none, none)
// CHECK: %[[NESTED_STORE:.*]] = dataflow.store {{.*}} %[[NESTED_CTRL]]#1
// CHECK: scf.if %arg5
// CHECK: dataflow.mux %arg1, %[[NESTED_CTRL]]#0, %[[NESTED_STORE]] : (i1, none, none) -> none
// CHECK: }
// CHECK: %[[TOP_DONE:.*]] = dataflow.mux %arg1, %[[NESTED_CTRL]]#0, %[[NESTED_STORE]] : (i1, none, none) -> none
// CHECK: dataflow.sync %[[TOP_DONE]] : (none) -> none
dataflow.graph.func private @g_nested_predicated_store_done(
    %arg0: none, %arg1: i1, %arg2: memref<?xi32>, %arg3: index,
    %arg4: i32, %arg5: i1) -> none {
  %ctrl_lanes:2 = dataflow.demux %arg1, %arg0
      : (i1, none) -> (none, none)
  %store_done = dataflow.store %arg2[%arg3] %arg4 %ctrl_lanes#1
      : memref<?xi32>
  scf.if %arg5 {
    %nested_done = dataflow.mux %arg1, %ctrl_lanes#0, %store_done
        : (i1, none, none) -> none
  }
  dataflow.graph.return %arg0 : none
}

// Multiple exact top-level completion muxes are ambiguous. The graph is left
// unchanged instead of adding another mux and sync.

// CHECK-LABEL: dataflow.graph.func private @g_duplicate_predicated_store_done
// CHECK: %[[DUP_CTRL:.*]]:2 = dataflow.demux %arg1, %arg0 : (i1, none) -> (none, none)
// CHECK: %[[DUP_STORE:.*]] = dataflow.store {{.*}} %[[DUP_CTRL]]#1
// CHECK: dataflow.mux %arg1, %[[DUP_CTRL]]#0, %[[DUP_STORE]] : (i1, none, none) -> none
// CHECK: dataflow.mux %arg1, %[[DUP_CTRL]]#0, %[[DUP_STORE]] : (i1, none, none) -> none
// CHECK-NOT: dataflow.sync
// CHECK: dataflow.graph.return %arg0
dataflow.graph.func private @g_duplicate_predicated_store_done(
    %arg0: none, %arg1: i1, %arg2: memref<?xi32>, %arg3: index,
    %arg4: i32) -> none {
  %ctrl_lanes:2 = dataflow.demux %arg1, %arg0
      : (i1, none) -> (none, none)
  // expected-remark@+1 {{loom-lower-graph-sync: multiple conditional store completion muxes match this store; leaving graph unchanged}}
  %store_done = dataflow.store %arg2[%arg3] %arg4 %ctrl_lanes#1
      : memref<?xi32>
  %done0 = dataflow.mux %arg1, %ctrl_lanes#0, %store_done
      : (i1, none, none) -> none
  %done1 = dataflow.mux %arg1, %ctrl_lanes#0, %store_done
      : (i1, none, none) -> none
  dataflow.graph.return %arg0 : none
}
