// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYNTH --implicit-check-not=unrealized_conversion_cast
// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost.yaml --print-stats=false --print-wallclock=false %s \
// RUN:   | loom - -loom-enumerate-fu-subgraphs > %t.enum.mlir
// RUN: grep -c "func.func private @fu0_subgraph_" %t.enum.mlir \
// RUN:   | FileCheck %s --check-prefix=COUNT
// RUN: FileCheck %s --check-prefix=ENUM < %t.enum.mlir

// The shared entry op is non-commutative and the second input reverses
// the subgraph block-argument order. Graph-MCS must infer the argument
// permutation before it can share that entry op around the private island.

// SYNTH: remark: {{.*}}synth-stat group=block_arg_perm_common_private strategy=mcs reason=success
// SYNTH-SAME: covered=2/2
// SYNTH-SAME: nodes=3/1/2
// SYNTH: fabric.fu
// SYNTH: fabric.op [@arith.subi]
// SYNTH-COUNT-2: fabric.demux
// SYNTH: fabric.op [@arith.muli]
// SYNTH: fabric.mux
// SYNTH: fabric.op [@arith.addi]

// COUNT: 2

// ENUM: func.func private @fu0_subgraph_0
// ENUM: arith.subi
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_1
// ENUM: arith.subi
// ENUM: arith.muli
// ENUM: arith.addi
// ENUM-NOT: func.func private @fu0_subgraph_2

func.func @pat_perm_direct(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "block_arg_perm_common_private"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                         %z = %c : i32) -> i32 {
    %diff = arith.subi %x, %y : i32
    %out = arith.addi %diff, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_perm_private(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "block_arg_perm_common_private"} {
  %r = dataflow.subgraph(%y = %b : i32, %x = %a : i32,
                         %z = %c : i32) -> i32 {
    %diff = arith.subi %x, %y : i32
    %p = arith.muli %diff, %z : i32
    %out = arith.addi %p, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
