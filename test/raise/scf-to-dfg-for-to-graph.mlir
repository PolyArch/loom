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

// Unused loop results must not become graph user-data returns. Pointer-walking
// loops often carry source/destination pointers only to drive memory accesses;
// if the enclosing thread does not use the final pointers, exposing them as
// graph results forces downstream mapping to model fake live pointer outputs.
// CHECK-LABEL: dataflow.thread private @t_unused_ptr_walk
// CHECK: dataflow.graph.launch @g_t_unused_ptr_walk_0(%{{.*}}) : (none, index, index, index, !llvm.ptr, !llvm.ptr) -> none
// CHECK-LABEL: func.func @host_reduction
// CHECK: dataflow.thread.launch @t_host_reduction_red_0
// CHECK-LABEL: dataflow.graph.func private @g_t_unused_ptr_walk_0
// CHECK-SAME: -> none
// CHECK: dataflow.graph.return %{{.*}} : none
dataflow.thread private @t_unused_ptr_walk(%src: !llvm.ptr, %dst: !llvm.ptr,
                                           %n: index) ctrl (%c: none) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %unused:2 = scf.for %i = %c0 to %n step %c1 iter_args(%s = %src, %d = %dst)
      -> (!llvm.ptr, !llvm.ptr) {
    %s_next = llvm.getelementptr %s[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %d_next = llvm.getelementptr %d[4] : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %s_next, %d_next : !llvm.ptr, !llvm.ptr
  }
  dataflow.thread.yield
}

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

// CHECK: dataflow.graph.func private @g_t_host_reduction_red_0_0
// CHECK-SAME: -> (none, f32)
// CHECK: dataflow.graph.return %{{.*}}, %{{.*}} : none, f32
