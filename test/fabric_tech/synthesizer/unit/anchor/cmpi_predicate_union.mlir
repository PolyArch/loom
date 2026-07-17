// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Two functions have identical topology but distinct comparison predicates.
// The shared physical op owns both complete typed modes, and the FU exposes
// one explicit encoding for each mode without field-wise recombination.

// CHECK: remark: {{.*}}synth-stat group=cmpi_pred strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=2
// CHECK: fabric.module @fu_cmpi_pred
// CHECK-SAME: loom.synthesized_for = "cmpi_pred"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.cmpi]
// CHECK-SAME: hw_params = [
// CHECK: fabric.yield

func.func @pat_cmpi_eq(%a: i32, %b: i32) -> i1
    attributes {loom.synth_group = "cmpi_pred"} {
  %s = arith.cmpi eq, %a, %b : i32
  return %s : i1
}

func.func @pat_cmpi_ne(%a: i32, %b: i32) -> i1
    attributes {loom.synth_group = "cmpi_pred"} {
  %s = arith.cmpi ne, %a, %b : i32
  return %s : i1
}
