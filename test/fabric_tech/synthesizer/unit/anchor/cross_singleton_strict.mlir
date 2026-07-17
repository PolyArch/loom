// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A with two distinct singleton ops at the same anchor position
// (math.absf and math.tan -- both 1-input fN -> fN, neither belongs to
// any multi-member hardware-share group). Default anchor config has
// `allow_intra_position_mux=false`, so the strategy must fail with
// `cross_share_group`. Each singleton occupies its own bucket: pre-fix,
// both singletons collapsed under one std::nullopt key and the path
// silently produced an op_list with two distinct singleton entries
// (which OpOp::verify rejects). Post-fix the strict path reports
// cross_share_group up-front and no IR is emitted.

// CHECK: warning:
// CHECK-SAME: group "fpu_unary_32_x": synthesis failed: cross_share_group
// CHECK: remark: {{.*}}synth-stat group=fpu_unary_32_x strategy=anchor reason=cross_share_group cost=0.000000e+00 covered=0/2 nodes=0/0/0
// CHECK: loom.synth_failed = "cross_share_group"
// CHECK: loom.synth_failed = "cross_share_group"

func.func @pat_absf(%a: f32) -> f32
    attributes {loom.synth_group = "fpu_unary_32_x"} {
  %s = math.absf %a : f32
  return %s : f32
}

func.func @pat_tan(%a: f32) -> f32
    attributes {loom.synth_group = "fpu_unary_32_x"} {
  %s = math.tan %a : f32
  return %s : f32
}
