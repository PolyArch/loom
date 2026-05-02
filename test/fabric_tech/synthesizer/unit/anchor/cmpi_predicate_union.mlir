// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A: two subgraphs of identical topology (yield <- arith.cmpi of two
// block args), one with predicate `eq`, the other with predicate `ne`.
// Per spec "hw_params policy" the synthesized FU's hw_params must surface
// the observed-value union of predicate strings -- otherwise the
// enumerator would not fan out the predicate axis and coverage
// verification would fail.

// CHECK: remark: {{.*}}synth-stat group=cmpi_pred strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: func.func @fu_cmpi_pred
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.cmpi]
// CHECK-SAME: hw_params = [{predicate = ["eq", "ne"]}]
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
