// RUN: loom-raise-opt --loom-lower-forall-to-thread %s | FileCheck %s

// Top-level scf.forall in a func.func body lowers to a sibling
// dataflow.thread definition + a dataflow.thread.launch at the
// original site. The forall's induction variable becomes a trailing
// `iv` block argument on the thread; captured outside-defined values
// (here %buf and %f0) flow as launch body operands.
//
// Two foralls in the same function get distinct sequential symbol
// names: t_<func>_0 and t_<func>_1.

// CHECK-LABEL: func.func @parallel_init
// CHECK: dataflow.thread.launch @t_parallel_init_0
// CHECK-NOT: scf.forall
func.func @parallel_init(%buf: memref<?xf32>, %n: index) {
    %f0 = arith.constant 0.0 : f32
    scf.forall (%i) in (%n) {
      memref.store %f0, %buf[%i] : memref<?xf32>
    }
    return
}

// CHECK-LABEL: func.func @two_foralls
// CHECK: dataflow.thread.launch @t_two_foralls_0
// CHECK: dataflow.thread.launch @t_two_foralls_1
func.func @two_foralls(%a: memref<?xf32>, %b: memref<?xf32>, %n: index) {
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    scf.forall (%i) in (%n) {
      memref.store %f0, %a[%i] : memref<?xf32>
    }
    scf.forall (%j) in (%n) {
      memref.store %f1, %b[%j] : memref<?xf32>
    }
    return
}

// All thread defs land at module scope after the func.func bodies.
// Each thread carries the spec-mandated thread_ctrl + iv slots
// (per spec section 5.4.1) on its entry block.
// CHECK-DAG: dataflow.thread private @t_parallel_init_0(%{{.*}}) ctrl (%{{.*}}: none) iv (%{{.*}}: index)
// CHECK-DAG: dataflow.thread private @t_two_foralls_0(%{{.*}}) ctrl (%{{.*}}: none) iv (%{{.*}}: index)
// CHECK-DAG: dataflow.thread private @t_two_foralls_1(%{{.*}}) ctrl (%{{.*}}: none) iv (%{{.*}}: index)
