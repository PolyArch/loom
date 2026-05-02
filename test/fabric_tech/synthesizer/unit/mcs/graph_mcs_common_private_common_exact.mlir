// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYNTH --implicit-check-not=unrealized_conversion_cast
// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost.yaml --print-stats=false --print-wallclock=false %s \
// RUN:   | loom - -loom-enumerate-fu-subgraphs \
// RUN:   | FileCheck %s --check-prefix=ENUM

// A graph-MCS candidate should share both common nodes around the private
// middle island. Existing positional shared-prefix matching only sees the
// leading addi and cannot share the final subi.

// SYNTH: remark: {{.*}}synth-stat group=common_private_common strategy=mcs reason=success
// SYNTH-SAME: covered=2/2
// SYNTH-SAME: nodes=3/1/1
// SYNTH: fabric.fu
// SYNTH: fabric.op [@arith.addi]
// SYNTH: fabric.demux
// SYNTH: fabric.op [@arith.muli]
// SYNTH: fabric.mux
// SYNTH: fabric.op [@arith.subi]

// ENUM: func.func private @fu0_subgraph_0
// ENUM: arith.addi
// ENUM: arith.subi
// ENUM: func.func private @fu0_subgraph_1
// ENUM: arith.addi
// ENUM: arith.muli
// ENUM: arith.subi
// ENUM-NOT: func.func private @fu0_subgraph_2

func.func @pat_direct(%a: i32, %b: i32, %d: i32) -> i32
    attributes {loom.synth_group = "common_private_common"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                         %w = %d : i32) -> i32 {
    %u = arith.addi %x, %y : i32
    %v = arith.subi %u, %w : i32
    dataflow.yield %v : i32
  }
  return %r : i32
}

func.func @pat_private_middle(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "common_private_common"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                         %z = %c : i32) -> i32 {
    %u = arith.addi %x, %y : i32
    %p = arith.muli %u, %z : i32
    %v = arith.subi %p, %z : i32
    dataflow.yield %v : i32
  }
  return %r : i32
}
