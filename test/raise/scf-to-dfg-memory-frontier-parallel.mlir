// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// CHECK-LABEL: dataflow.graph private @repeat_parallel(
// CHECK: dataflow.carry
// CHECK: %[[REPEAT_LOAD0:.*]], %[[REPEAT_DONE0:.*]] = dataflow.load
// CHECK: %[[REPEAT_READ0:.*]]:2 = dataflow.sync {{.*}}%[[REPEAT_DONE0]]
// CHECK: %[[REPEAT_LOAD1:.*]], %[[REPEAT_DONE1:.*]] = dataflow.load
// CHECK: %[[REPEAT_READ1:.*]]:2 = dataflow.sync {{.*}}%[[REPEAT_DONE1]]
// CHECK: dataflow.sync %[[REPEAT_READ0]]#0, %[[REPEAT_READ1]]#0
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return
dataflow.graph private @repeat_parallel(
    %start: none, %limit: index, %memory: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %outer = %c0 to %limit step %c1 {
    scf.parallel (%lane) = (%c0) to (%c2) step (%c1) {
      %index = arith.addi %outer, %lane : index
      %value = memref.load %memory[%index] : memref<?xi32>
      scf.reduce
    }
  }
  dataflow.graph.return %start : none
}

// Branch-local parallel joins must precede the outer selected frontier.
// CHECK-LABEL: dataflow.graph private @select_parallel(
// CHECK: %[[TRUE_DONE0:.*]] = dataflow.store
// CHECK: %[[TRUE_DONE1:.*]] = dataflow.store
// CHECK: dataflow.sync %[[TRUE_DONE0]], %[[TRUE_DONE1]]
// CHECK: dataflow.mux
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return
dataflow.graph private @select_parallel(
    %start: none, %condition: i1, %memory: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %value = arith.constant 7 : i32
  scf.if %condition {
    scf.parallel (%lane) = (%c0) to (%c2) step (%c1) {
      memref.store %value, %memory[%lane] : memref<?xi32>
      scf.reduce
    }
  }
  dataflow.graph.return %start : none
}

// Each lane selects its own frontier before the fixed-width group joins.
// CHECK-LABEL: dataflow.graph private @parallel_select(
// CHECK: dataflow.mux
// CHECK: dataflow.mux
// CHECK: dataflow.sync
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return
dataflow.graph private @parallel_select(
    %start: none, %condition: i1, %memory: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%lane) = (%c0) to (%c2) step (%c1) {
    scf.if %condition {
      %value = memref.load %memory[%lane] : memref<?xi32>
    }
    scf.reduce
  }
  dataflow.graph.return %start : none
}

// Each fixed parallel level proves disjointness in its own induction
// dimensions. Descendant lane coordinates remain independent during an
// ancestor proof.
// CHECK-LABEL: dataflow.graph private @nested_parallel_matrix(
// CHECK-COUNT-4: dataflow.store
// CHECK-NOT: scf.
// CHECK: dataflow.graph.return
dataflow.graph private @nested_parallel_matrix(
    %start: none, %memory: memref<2x2xi32>) -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %value = arith.constant 7 : i32
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    scf.parallel (%j) = (%c0) to (%c2) step (%c1) {
      memref.store %value, %memory[%i, %j] : memref<2x2xi32>
      scf.reduce
    }
    scf.reduce
  }
  dataflow.graph.return %start : none
}
