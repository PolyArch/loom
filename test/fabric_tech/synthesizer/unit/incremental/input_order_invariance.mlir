// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A inputs presented in two different lexical orders: group g1 has
// inputs in (addi, subi) order; group g2 has inputs in (subi, addi)
// order. With largest_first heuristic both groups should produce
// identical FUs (since both subgraphs have one body op the heuristic
// falls back to lexical func name as a tie-breaker, which is the same
// across both groups).

// CHECK: remark: {{.*}}synth-stat group=g1 strategy=incremental reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: remark: {{.*}}synth-stat group=g2 strategy=incremental reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: fabric.module @fu_g1
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK: fabric.module @fu_g2
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]

func.func @g1_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "g1"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @g1_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "g1"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @g2_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "g2"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @g2_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "g2"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
