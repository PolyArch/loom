// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/../unit/anchor/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// Two func.funcs with no `loom.synth_group` attribute land in the implicit
// `default` group. Anchor folds both inputs into one FU and emits one real
// coverage statistic for the group.

// CHECK: remark:
// CHECK-SAME: synth-stat group=default strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0 encodings=2
// CHECK: fabric.module @fu_default
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]

func.func @pat_addi(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
func.func @pat_subi(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
