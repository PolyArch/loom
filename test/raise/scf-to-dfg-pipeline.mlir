// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// The combined --loom-lower-scf-to-dfg pipeline runs forall-to-thread
// and graph extraction in sequence. Parallel work is exposed through
// a thread and graph. Host reductions remain in SCF until a real
// accelerator-region promotion owns their execution semantics.

// CHECK-LABEL: func.func @vecadd_like
// CHECK: dataflow.thread.launch @t_vecadd_like_0
// CHECK-NOT: dataflow.thread.launch @t_vecadd_like_red
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

func.func @nested_forall_reduction(%out: memref<?xi32>, %n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> (index) {
    %end = arith.addi %acc, %c1 : index
    %active = arith.cmpi ult, %acc, %end : index
    scf.if %active {
      scf.forall (%j) = (%acc) to (%end) step (1) {
        %v = arith.index_cast %j : index to i32
        memref.store %v, %out[%j] : memref<?xi32>
      }
    }
    scf.yield %end : index
  }
  return %sum : index
}

// CHECK: dataflow.thread private @t_vecadd_like_0
// CHECK-SAME: ctrl (%{{.*}}: none) iv (%{{.*}}: index)
// CHECK: dataflow.graph.launch @g_t_vecadd_like_0_0
// CHECK-LABEL: dataflow.graph private @g_t_vecadd_like_0_0
// The graph.launch dependency is the enclosing thread's thread_ctrl
// block argument; the lowered IR contains no ub.poison.
// CHECK-NOT: ub.poison : none
// CHECK-NOT: _red_
