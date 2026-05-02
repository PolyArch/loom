// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=fabric.mux --implicit-check-not=fabric.demux

// Two identical two-node DAGs are already a complete common skeleton.
// The best FU shares both operators directly without selection hardware.

// CHECK: remark: {{.*}}synth-stat group=identical_two_node strategy=mcs reason=success
// CHECK-SAME: cost=2.000000e+00
// CHECK-SAME: covered=2/2
// CHECK-SAME: nodes=2/0/0
// CHECK: fabric.module @fu_identical_two_node
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi]
// CHECK: fabric.op [@arith.muli]
// CHECK: fabric.yield

func.func @pat_add_then_mul_a(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "identical_two_node"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.muli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_add_then_mul_b(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "identical_two_node"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.muli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
