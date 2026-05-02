// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true' 2>&1 | FileCheck %s

// Two func.funcs with no `loom.synth_group` attribute land in the implicit
// `default` group. The default `incremental_random` strategy folds both
// inputs into a single shared FU; the `dump-stats=true` flag emits one
// canonical `synth-stat` line per group.

// CHECK: remark:
// CHECK-SAME: synth-stat group=default strategy=incremental_random reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: func.func @fu_default
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
