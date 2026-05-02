// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/incremental.yaml dump-stats=true' 2>&1 | FileCheck %s

// Tier B: two inputs share an arith.addi prefix; one extends with
// arith.muli at the yield, one terminates immediately. The incremental
// strategy starts from the trivial FU for input_0 (just arith.addi)
// then folds input_1 by inserting a fabric.demux on the addi output
// (one arm to yield, one arm to muli) and a fabric.mux at the yield
// to collapse both branches back into the single output port.

// CHECK: remark: {{.*}}synth-stat group=tierB_demo strategy=incremental reason=success
// CHECK: func.func @fu_tierB_demo
// CHECK: fabric.fu
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.demux
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK-DAG: fabric.mux
// CHECK: fabric.yield

func.func @pat_add_only(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "tierB_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    dataflow.yield %t : i32
  }
  return %r : i32
}

func.func @pat_add_then_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "tierB_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    %m = arith.muli %t, %z : i32
    dataflow.yield %m : i32
  }
  return %r : i32
}
