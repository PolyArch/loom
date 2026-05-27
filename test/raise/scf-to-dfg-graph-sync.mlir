// RUN: loom-raise-opt --loom-lower-graph-sync %s | FileCheck %s

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
      {cont_cond = "<", step_op = "+="} : i64
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
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.invariant %rwc, %arg4 : f32
  dataflow.graph.return %arg0, %0 : none, f32
}
