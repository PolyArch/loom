// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// The combined --loom-lower-scf-to-dfg pipeline runs forall-to-thread
// and for-to-graph in sequence. A function with a parallel-init forall
// followed by a reduction tail emits two threads (one per forall, one
// for the wrapped reduction) plus one graph (the reduction body).

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

// CHECK: dataflow.thread private @t_vecadd_like_0
// CHECK-SAME: ctrl (%{{.*}}: none) iv (%{{.*}}: index)
// CHECK: dataflow.thread private @t_vecadd_like_red_0
// CHECK-SAME: ctrl (%{{.*}}: none) iv (%{{.*}}: index)
// CHECK: dataflow.graph.func private @g_t_vecadd_like_red_0_0
// The graph.launch's ctrl_in is the enclosing thread's thread_ctrl
// block argument; the lowered IR contains no ub.poison.
// CHECK-NOT: ub.poison : none
