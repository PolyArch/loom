// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A: two subgraphs that both use the SAME singleton op (math.absf,
// not in any multi-member hardware-share group). Both peers map to one
// singleton bucket whose op_list has the single entry @math.absf, so
// the strategy emits one fabric.op (no mux) and OpOp::verify accepts
// the single-name op_list. This pins the regression: the bucket key
// for matching singletons must collapse, while the bucket key for two
// distinct singletons (covered by `cross_singleton_*.mlir`) must not.

// CHECK: remark: {{.*}}synth-stat group=fpu_abs_32 strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: func.func @fu_fpu_abs_32
// CHECK: fabric.fu
// CHECK: fabric.op [@math.absf]
// CHECK-SAME: hw_params = [{}]
// CHECK: !fabric.bits<32>
// CHECK: fabric.yield

func.func @pat_absf_a(%a: f32) -> f32
    attributes {loom.synth_group = "fpu_abs_32"} {
  %r = dataflow.subgraph(%x = %a : f32) -> f32 {
    %s = math.absf %x : f32
    dataflow.yield %s : f32
  }
  return %r : f32
}

func.func @pat_absf_b(%a: f32) -> f32
    attributes {loom.synth_group = "fpu_abs_32"} {
  %r = dataflow.subgraph(%x = %a : f32) -> f32 {
    %s = math.absf %x : f32
    dataflow.yield %s : f32
  }
  return %r : f32
}
