// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/order_heuristic_largest_first.yaml dump-stats=true' 2>&1 | FileCheck %s

// Wire test for `synth.incremental_random.input_order_heuristic`. With
// `largest_first`, the first restart's permutation is the deterministic
// largest_first ordering (descending body-node count, lex-by-funcname
// tiebreaker); the remaining `restarts - 1` permutations stay random.
// `restarts: 1` makes the heuristic-driven permutation the sole input
// ordering exercised, so this test verifies that the new heuristic is
// read from the config and consumed by the strategy.
//
// The inputs are tier-A friendly (every subgraph is a single body op
// drawn from the arith.addi/arith.subi share group), so synthesis must
// succeed regardless of permutation; the wrapper carries the
// share-aware widened op_list. The largest_first heuristic still uses
// the lexical func-name tiebreaker for equal-sized bodies, which is
// the path the wire test exercises.

// CHECK: remark: {{.*}}synth-stat group=lf_demo strategy=incremental_random reason=success
// CHECK: func.func @fu_lf_demo
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK: fabric.yield

func.func @lf_a_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "lf_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @lf_b_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "lf_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
