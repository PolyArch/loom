// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// A counted loop streams under its own comparison domain: an unsigned loop
// compares unsigned and a signed loop compares signed.
// CHECK-LABEL: dataflow.graph private @unsigned_counted
// CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step add while ult
dataflow.graph private @unsigned_counted(%ctrl: none, %n: i64) -> i64
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r = scf.for unsigned %iv = %c0 to %n step %c1 iter_args(%acc = %c0) -> (i64) : i64 {
    %next = arith.addi %acc, %iv : i64
    scf.yield %next : i64
  }
  dataflow.graph.return %ctrl, %r : none, i64
}

// CHECK-LABEL: dataflow.graph private @signed_counted
// CHECK: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} step add while slt
dataflow.graph private @signed_counted(%ctrl: none, %n: i64) -> i64
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 1, 0, 0>} {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %c0) -> (i64) : i64 {
    %next = arith.addi %acc, %iv : i64
    scf.yield %next : i64
  }
  dataflow.graph.return %ctrl, %r : none, i64
}
