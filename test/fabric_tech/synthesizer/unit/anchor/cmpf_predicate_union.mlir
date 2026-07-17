// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Two functions have identical topology but distinct comparison predicates.
// The shared physical op owns both complete typed modes, and the FU exposes
// one explicit encoding for each mode without field-wise recombination.

// CHECK: remark: {{.*}}synth-stat group=cmpf_pred strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=2
// CHECK: fabric.module @fu_cmpf_pred
// CHECK-SAME: loom.synthesized_for = "cmpf_pred"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.cmpf]
// CHECK-SAME: hw_params = [
// CHECK: fabric.yield

func.func @pat_cmpf_oeq(%a: f32, %b: f32) -> i1
    attributes {loom.synth_group = "cmpf_pred"} {
  %s = arith.cmpf oeq, %a, %b : f32
  return %s : i1
}

func.func @pat_cmpf_one(%a: f32, %b: f32) -> i1
    attributes {loom.synth_group = "cmpf_pred"} {
  %s = arith.cmpf one, %a, %b : f32
  return %s : i1
}
