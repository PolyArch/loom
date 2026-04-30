// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/order_heuristic_smallest_first.yaml dump-stats=true' 2>&1 | FileCheck %s

// Wire test for `synth.incremental_random.input_order_heuristic` with
// the `smallest_first` value. With `restarts: 1`, the first (and only)
// restart's permutation is the deterministic smallest_first ordering
// (ascending body-node count, lex-by-funcname tiebreaker), so the test
// verifies that the smallest_first path is read from the config and
// consumed by the strategy. The inputs are tier-A friendly so synthesis
// must succeed and produce the share-aware widened op_list.

// CHECK: remark: {{.*}}synth-stat group=sf_demo strategy=incremental_random reason=success
// CHECK: func.func @fu_sf_demo
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK: fabric.yield

func.func @sf_a_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sf_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @sf_b_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sf_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
