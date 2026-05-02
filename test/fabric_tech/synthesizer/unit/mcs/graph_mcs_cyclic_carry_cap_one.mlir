// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_cap_one.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// With candidate_cap=1, MCS cannot rely on compatibility restarts after
// local graph-MCS search. A cyclic graph-region body with a carry backedge
// must still synthesize through a verified local candidate.

// CHECK: remark: {{.*}}synth-stat group=cyclic_carry_cap_one strategy=mcs reason=success
// CHECK-SAME: covered=2/2
// CHECK-SAME: nodes=2/0/0
// CHECK: fabric.fu
// CHECK: fabric.op [@dataflow.carry]
// CHECK: fabric.op [@arith.addi]

func.func @pat_carry_a(%cond: i1, %init: i32, %step: i32) -> i32
    attributes {loom.synth_group = "cyclic_carry_cap_one"} {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32,
                         %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}

func.func @pat_carry_b(%cond: i1, %init: i32, %step: i32) -> i32
    attributes {loom.synth_group = "cyclic_carry_cap_one"} {
  %r = dataflow.subgraph(%c = %cond : i1, %i = %init : i32,
                         %s = %step : i32) -> i32 {
    %acc = dataflow.carry %c, %i, %next : i32
    %next = arith.addi %acc, %s : i32
    dataflow.yield %acc : i32
  }
  return %r : i32
}
