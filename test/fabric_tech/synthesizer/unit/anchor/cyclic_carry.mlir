// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// CHECK: remark: {{.*}}synth-stat group=cyclic_carry strategy=anchor reason=success
// CHECK-SAME: covered=2/2
// CHECK-DAG: fabric.op [@dataflow.carry]
// CHECK-DAG: fabric.op [@arith.addi]

func.func @carry_a(%cond: i1, %init: i32, %step: i32) -> i32
    attributes {loom.synth_group = "cyclic_carry"} {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32,
                         %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}

func.func @carry_b(%cond: i1, %init: i32, %step: i32) -> i32
    attributes {loom.synth_group = "cyclic_carry"} {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32,
                         %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}
