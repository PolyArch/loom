// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost.yaml --print-stats=true --print-wallclock=false %s > %t.fu.mlir
// RUN: FileCheck %s --check-prefix=SYNTH --implicit-check-not=unrealized_conversion_cast < %t.fu.mlir
// RUN: sed '/^synth-stat /d' %t.fu.mlir | loom - -loom-enumerate-fu-subgraphs > %t.enum.mlir
// RUN: grep -c "func.func private @fu0_subgraph_" %t.enum.mlir \
// RUN:   | FileCheck %s --check-prefix=COUNT
// RUN: FileCheck %s --check-prefix=ENUM < %t.enum.mlir

// Three acyclic inputs share the entry add and final subtract while the
// private middle region has zero, one, or two body ops. Enumeration should
// recover exactly the three source shapes.

// SYNTH: fabric.fu
// SYNTH: fabric.op [@arith.addi]
// SYNTH: fabric.demux
// SYNTH: fabric.op [@arith.muli]
// SYNTH: fabric.op [@arith.xori]
// SYNTH: fabric.mux
// SYNTH: fabric.op [@arith.subi]
// SYNTH: synth-stat group=acyclic_private_sizes strategy=mcs reason=success
// SYNTH-SAME: covered=3/3

// COUNT: 3

// ENUM: func.func private @fu0_subgraph_0
// ENUM: arith.addi
// ENUM: arith.subi
// ENUM: func.func private @fu0_subgraph_1
// ENUM: arith.addi
// ENUM: arith.muli
// ENUM: arith.subi
// ENUM: func.func private @fu0_subgraph_2
// ENUM: arith.addi
// ENUM: arith.muli
// ENUM: arith.xori
// ENUM: arith.subi
// ENUM-NOT: func.func private @fu0_subgraph_3

func.func @pat_private_zero(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "acyclic_private_sizes"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                         %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.subi %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_private_one(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "acyclic_private_sizes"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                         %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %p = arith.muli %sum, %z : i32
    %out = arith.subi %p, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_private_two(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "acyclic_private_sizes"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                         %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %p = arith.muli %sum, %z : i32
    %q = arith.xori %p, %z : i32
    %out = arith.subi %q, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
