// RUN: loom-raise-opt --loom-lower-for-to-graph %s | FileCheck %s

// CHECK-LABEL: dataflow.thread private @t_existing
// CHECK-SAME: ctrl (%[[CTRL:.*]]: none)
// CHECK: %{{.*}}, %[[DONE:.*]] = dataflow.graph.launch @g_t_existing_0 deps(%[[CTRL]])
// CHECK: dataflow.thread.yield %[[DONE]] : none
// CHECK-NOT: ub.poison : none
// CHECK-NOT: scf.for {{.*}} iter_args

// CHECK-LABEL: dataflow.thread private @t_straight
// CHECK-SAME: ctrl (%[[STRAIGHT_CTRL:.*]]: none)
// CHECK: %[[STRAIGHT_DONE:.*]] = dataflow.graph.launch @g_t_straight_0 deps(%[[STRAIGHT_CTRL]])
// CHECK: dataflow.thread.yield %[[STRAIGHT_DONE]] : none

// CHECK-LABEL: func.func @host_reduction
// CHECK-NOT: dataflow.thread.launch @t_host_reduction
// CHECK: scf.for {{.*}} iter_args

// CHECK-LABEL: dataflow.graph private @g_t_existing_0
// CHECK-NOT: scf.
// CHECK: dataflow.stream
// CHECK: dataflow.load
// CHECK: dataflow.graph.return

dataflow.thread private @t_existing(
    %buf: memref<?xf32>, %n: index) ctrl (%ctrl: none) {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %index = %c0 to %n step %c1
      iter_args(%acc = %f0) -> (f32) {
    %value = memref.load %buf[%index] : memref<?xf32>
    %next = arith.addf %acc, %value : f32
    scf.yield %next : f32
  }
  dataflow.thread.yield
}

dataflow.thread private @t_straight(%value: i32) ctrl (%ctrl: none) {
  %sum = arith.addi %value, %value : i32
  dataflow.thread.yield
}

func.func @host_reduction(%buf: memref<?xf32>, %n: index) -> f32 {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %index = %c0 to %n step %c1
      iter_args(%acc = %f0) -> (f32) {
    %value = memref.load %buf[%index] : memref<?xf32>
    %next = arith.addf %acc, %value : f32
    scf.yield %next : f32
  }
  return %result : f32
}
