// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Both functions use the same complete `dataflow.sync` mode. The shared
// physical op owns that typed mode once, and the FU needs one encoding.

// CHECK: remark: {{.*}}synth-stat group=sync_pair strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=1
// CHECK: fabric.module @fu_sync_pair
// CHECK-SAME: loom.synthesized_for = "sync_pair"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.sync]
// CHECK-SAME: hw_params = [
// CHECK: fabric.yield

func.func @pat_sync_a(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sync_pair"} {
  %u, %v = dataflow.sync %a, %b : (i32, i32) -> (i32, i32)
  return %u : i32
}

func.func @pat_sync_b(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sync_pair"} {
  %u, %v = dataflow.sync %a, %b : (i32, i32) -> (i32, i32)
  return %u : i32
}
