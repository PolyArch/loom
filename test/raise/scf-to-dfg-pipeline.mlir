// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// The combined --loom-lower-scf-to-dfg pipeline runs forall-to-thread
// and graph extraction in sequence. A function with a parallel-init
// forall followed by a reduction tail emits one graph for the
// straight-line parallel body and one graph for the reduction body.

// CHECK-LABEL: func.func @vecadd_like
// CHECK: dataflow.thread.launch @t_vecadd_like_0
// CHECK: dataflow.thread.launch @t_vecadd_like_red_0
func.func @vecadd_like(%a: memref<?xf32>, %b: memref<?xf32>, %n: index) -> f32 {
  scf.forall (%i) in (%n) {
    %v = memref.load %a[%i] : memref<?xf32>
    %v2 = arith.mulf %v, %v : f32
    memref.store %v2, %b[%i] : memref<?xf32>
  }
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> (f32) {
    %v = memref.load %b[%i] : memref<?xf32>
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  return %r : f32
}

func.func @mean_like(%a: memref<?xf32>, %n: i64) -> f32 {
  %f0 = arith.constant 0.0 : f32
  %scale = arith.constant 0.015625 : f32
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> (f32) : i64 {
    %idx = arith.index_cast %i : i64 to index
    %v = memref.load %a[%idx] : memref<?xf32>
    %next = arith.addf %acc, %v : f32
    scf.yield %next : f32
  }
  %mean = arith.mulf %sum, %scale : f32
  return %mean : f32
}

// CHECK: dataflow.thread private @t_vecadd_like_0
// CHECK-SAME: ctrl (%{{.*}}: none) iv (%{{.*}}: index)
// CHECK: dataflow.graph.launch @g_t_vecadd_like_0_0
// CHECK: dataflow.thread private @t_vecadd_like_red_0
// CHECK-SAME: ctrl (%{{.*}}: none) iv (%{{.*}}: index)
// CHECK: dataflow.graph.launch @g_t_vecadd_like_red_0_0
// CHECK: dataflow.graph.func private @g_t_vecadd_like_red_0_0
// CHECK-LABEL: dataflow.graph.func private @g_t_mean_like_red_0_0
// CHECK: %[[SCALE:.*]] = dataflow.invariant
// CHECK: %[[MEAN:.*]] = arith.mulf %{{.*}}, %[[SCALE]] : f32
// CHECK: dataflow.graph.return %{{.*}}, %[[MEAN]] : none, f32
// CHECK-LABEL: dataflow.graph.func private @g_t_vecadd_like_0_0
// The graph.launch's ctrl_in is the enclosing thread's thread_ctrl
// block argument; the lowered IR contains no ub.poison.
// CHECK-NOT: ub.poison : none
