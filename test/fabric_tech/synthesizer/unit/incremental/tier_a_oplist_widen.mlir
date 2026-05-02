// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier A: two subgraphs of identical topology (yield <- bin-op of two
// block args) sharing the arith.addi/subi hardware-share group. The
// incremental strategy starts from the trivial FU built from input_0
// (arith.addi) then folds input_1 (arith.subi) by widening op_list.
// The result must match the anchor strategy's output on the same input.

// CHECK: remark: {{.*}}synth-stat group=alu_int_32 strategy=incremental reason=success
// CHECK-SAME: covered=2/2 nodes=1/0/0
// CHECK: fabric.module @fu_alu_int_32
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi, @arith.subi]
// CHECK-SAME: hw_params = [{}]
// CHECK: !fabric.bits<32>
// CHECK: fabric.yield

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_subi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
