// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor_with_mux.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A with two distinct singleton ops at the same anchor position
// (math.absf and math.tan), with intra-position muxing enabled. Each
// distinct singleton occupies its own share-group bucket, so the
// strategy emits one fabric.op per singleton (each op_list has exactly
// one entry) and joins them through a fresh fabric.mux. The coverage
// verifier is left enabled (its default): the synthesized FU must
// round-trip through SubgraphEnumerator + SubgraphMatcher cleanly,
// including the fabric.mux output's float-flavor lift inference.

// CHECK: remark: {{.*}}synth-stat group=fpu_unary_32_x strategy=anchor reason=success
// CHECK: fabric.module @fu_fpu_unary_32_x
// CHECK-SAME: loom.synthesized_for = "fpu_unary_32_x"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@math.absf]
// CHECK-DAG: fabric.op [@math.tan]
// CHECK: fabric.mux

func.func @pat_absf(%a: f32) -> f32
    attributes {loom.synth_group = "fpu_unary_32_x"} {
  %r = dataflow.subgraph(%x = %a : f32) -> f32 {
    %s = math.absf %x : f32
    dataflow.yield %s : f32
  }
  return %r : f32
}

func.func @pat_tan(%a: f32) -> f32
    attributes {loom.synth_group = "fpu_unary_32_x"} {
  %r = dataflow.subgraph(%x = %a : f32) -> f32 {
    %s = math.tan %x : f32
    dataflow.yield %s : f32
  }
  return %r : f32
}
