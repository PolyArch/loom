// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

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
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i1 {
    %s = arith.cmpi eq, %x, %y : i32
    dataflow.yield %s : i1
  }
  return %r : i1
}

func.func @pat_cmpi_ne(%a: i32, %b: i32) -> i1
    attributes {loom.synth_group = "cmpi_pred"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i1 {
    %s = arith.cmpi ne, %x, %y : i32
    dataflow.yield %s : i1
  }
  return %r : i1
}
