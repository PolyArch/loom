// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A: two subgraphs of identical topology, each yielding the first
// output of a 2-input `dataflow.sync`. Both inputs have the same arity
// (M=2), so the observed-value union of the variadic `bitmask` axis
// collapses to a single value `"11"`. The synthesized FU's hw_params
// must surface that explicit allowed set (rather than `[{}]`) so the
// enumerator's bitmask fan-out is constrained to the observed set.

// CHECK: remark: {{.*}}synth-stat group=sync_pair strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: func.func @fu_sync_pair
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.sync]
// CHECK-SAME: hw_params = [{bitmask = ["11"]}]
// CHECK: fabric.yield

func.func @pat_sync_a(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sync_pair"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %u, %v = dataflow.sync %x, %y : (i32, i32) -> (i32, i32)
    dataflow.yield %u : i32
  }
  return %r : i32
}

func.func @pat_sync_b(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sync_pair"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %u, %v = dataflow.sync %x, %y : (i32, i32) -> (i32, i32)
    dataflow.yield %u : i32
  }
  return %r : i32
}
