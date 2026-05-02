// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_cap_one.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// Anchor cannot cover this cyclic pair because the second input inserts a
// private op inside the recurrence. With candidate_cap=1, the success must
// come from the local graph-MCS candidate and its forward-reference
// placeholder rewiring.

// CHECK: remark: {{.*}}synth-stat group=cyclic_carry_private_common strategy=mcs reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.carry]
// CHECK: fabric.demux
// CHECK: fabric.op [@arith.muli]
// CHECK: fabric.mux
// CHECK: fabric.op [@arith.addi]

func.func @pat_carry_direct(%cond: i1, %init: i32, %step: i32) -> i32
    attributes {loom.synth_group = "cyclic_carry_private_common"} {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32,
                         %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}

func.func @pat_carry_private(%cond: i1, %init: i32, %step: i32) -> i32
    attributes {loom.synth_group = "cyclic_carry_private_common"} {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32,
                         %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %p = arith.muli %acc, %s : i32
    %next = arith.addi %p, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}
