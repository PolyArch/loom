// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

// scf.for with iter_args inside a dataflow.thread body lowers to a
// sibling dataflow.graph.func definition + a dataflow.graph.launch
// at the cut site. The host_reduction case below also exercises the
// host-scope wrap path: a stand-alone reduction at host scope is
// wrapped in a synthetic 1x1 thread before being promoted.

// The thread carries the spec-mandated thread_ctrl slot, and the
// graph.launch consumes it directly as ctrl_in (no ub.poison).
// CHECK-LABEL: dataflow.thread private @t_existing
// CHECK-SAME: ctrl (%[[CTRL:.*]]: none)
// CHECK: dataflow.graph.launch @g_t_existing_0(%[[CTRL]]
// CHECK-NOT: ub.poison : none
// CHECK-NOT: scf.for {{.*}} iter_args
dataflow.thread private @t_existing(%buf: memref<?xf32>, %n: index) ctrl (%c: none) {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> (f32) {
    %v = memref.load %buf[%i] : memref<?xf32>
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  dataflow.thread.yield
}

// CHECK-LABEL: func.func @host_reduction
// CHECK: dataflow.thread.launch @t_host_reduction_red_0
func.func @host_reduction(%buf: memref<?xf32>, %n: index) -> f32 {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> (f32) {
    %v = memref.load %buf[%i] : memref<?xf32>
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  return %r : f32
}

// CHECK-DAG: dataflow.graph.func private @g_t_existing_0
// CHECK-DAG: dataflow.thread private @t_host_reduction_red_0
// CHECK-DAG: dataflow.graph.func private @g_t_host_reduction_red_0_0
