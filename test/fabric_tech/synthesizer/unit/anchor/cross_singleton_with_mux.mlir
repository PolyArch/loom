// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor_with_mux.yaml dump-stats=true' 2>&1 | FileCheck %s

// Distinct singleton ops share one FU input through an explicit demux and
// join their results through a mux. Each legal route is a complete encoding.

// CHECK: remark: {{.*}}synth-stat group=fpu_unary_32_x strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=2/1/1 encodings=2
// CHECK: fabric.module @fu_fpu_unary_32_x
// CHECK-SAME: loom.synthesized_for = "fpu_unary_32_x"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.demux
// CHECK-DAG: fabric.op [@math.absf]
// CHECK-DAG: fabric.op [@math.tan]
// CHECK: fabric.mux

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
