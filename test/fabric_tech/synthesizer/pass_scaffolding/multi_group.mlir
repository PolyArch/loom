// RUN: loom %s -loom-synthesize-configured-functions='config=%p/../unit/anchor/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Three func.funcs in two groups (`alu` x 2, `fpu` x 1). With
// `dump-stats=true` we expect one `synth-stat` remark per group, ordered
// lexically by group name (`alu` before `fpu`). Anchor synthesizes both
// groups with explicit semantic encodings.

// CHECK: remark: {{.*}}synth-stat group=alu strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=2
// CHECK: remark: {{.*}}synth-stat group=fpu strategy=anchor reason=success
// CHECK-SAME: covered=1/1 nodes=1/0/0 encodings=1

func.func @pat_alu_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "alu"} {
  %s = arith.addi %a, %b : i32
  return %s : i32
}
func.func @pat_alu_subi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "alu"} {
  %s = arith.subi %a, %b : i32
  return %s : i32
}
func.func @pat_fpu_addf(%a: f32, %b: f32) -> f32 attributes {loom.synth_group = "fpu"} {
  %s = arith.addf %a, %b : f32
  return %s : f32
}
