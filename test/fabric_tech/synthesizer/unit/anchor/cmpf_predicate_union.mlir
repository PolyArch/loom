// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A: two subgraphs of identical topology (yield <- arith.cmpf of two
// f32 block args), one with predicate `oeq`, the other with predicate
// `one`. Per spec "hw_params policy" the synthesized FU's hw_params must
// surface the observed-value union of predicate strings so the
// enumerator's `predicate` axis fan-out covers both inputs.

// CHECK: remark: {{.*}}synth-stat group=cmpf_pred strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: fabric.module @fu_cmpf_pred
// CHECK-SAME: loom.synthesized_for = "cmpf_pred"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.cmpf]
// CHECK-SAME: hw_params = [{predicate = ["oeq", "one"]}]
// CHECK: fabric.yield

func.func @pat_cmpf_oeq(%a: f32, %b: f32) -> i1
    attributes {loom.synth_group = "cmpf_pred"} {
  %r = dataflow.subgraph(%x = %a : f32, %y = %b : f32) -> i1 {
    %s = arith.cmpf oeq, %x, %y : f32
    dataflow.yield %s : i1
  }
  return %r : i1
}

func.func @pat_cmpf_one(%a: f32, %b: f32) -> i1
    attributes {loom.synth_group = "cmpf_pred"} {
  %r = dataflow.subgraph(%x = %a : f32, %y = %b : f32) -> i1 {
    %s = arith.cmpf one, %x, %y : f32
    dataflow.yield %s : i1
  }
  return %r : i1
}
