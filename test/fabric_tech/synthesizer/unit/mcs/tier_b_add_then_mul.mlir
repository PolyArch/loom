// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs.yaml dump-stats=true' 2>&1 | FileCheck %s

// Acceptance criterion 2 (mcs): on `(a+b)*c` and `(a+b)` mixed inputs,
// mcs identifies the shared `arith.addi` skeleton and bypasses the
// multiplication via a single `fabric.mux` (and a `fabric.demux` on
// the addi's downstream branches).
//
// Same input shape as the incremental strategy's tier-B test, so the
// strategies should converge on a structurally equivalent wrapper:
// addi -> demux -> {direct yield arm, muli arm} -> mux -> yield.

// CHECK: remark: {{.*}}synth-stat group=tierB_demo strategy=mcs reason=success
// CHECK: fabric.module @fu_tierB_demo
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.demux
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK-DAG: fabric.mux
// CHECK: fabric.yield

func.func @pat_add_only(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "tierB_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    dataflow.yield %t : i32
  }
  return %r : i32
}

func.func @pat_add_then_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "tierB_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    %m = arith.muli %t, %z : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}
