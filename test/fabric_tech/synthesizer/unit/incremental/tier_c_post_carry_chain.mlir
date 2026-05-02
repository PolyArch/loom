// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// Tier C: a differing post-carry operation may feed another operation
// before becoming the carry back-edge value.

// CHECK: remark: {{.*}}synth-stat group=post_carry_chain strategy=incremental reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_post_carry_chain
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.stream]
// CHECK: fabric.op [@dataflow.carry]
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.xori]
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK-DAG: fabric.mux
// CHECK-NOT: fabric.op [@dataflow.carry]
// CHECK: fabric.yield

func.func @pat_chain_addi(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "post_carry_chain"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %a = arith.addi %c, %idx : i32
    %nxt = arith.muli %a, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}

func.func @pat_chain_xori(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "post_carry_chain"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %x = arith.xori %c, %idx : i32
    %nxt = arith.muli %x, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}
