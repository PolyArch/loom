// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost_cap_one.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYNTH --implicit-check-not=unrealized_conversion_cast
// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost_cap_one.yaml --print-stats=false --print-wallclock=false %s \
// RUN:   | loom - -loom-enumerate-fu-subgraphs > %t.enum.mlir
// RUN: grep -c "func.func private @fu0_subgraph_" %t.enum.mlir \
// RUN:   | FileCheck %s --check-prefix=COUNT
// RUN: FileCheck %s --check-prefix=ENUM < %t.enum.mlir

// The inputs return two values from shared terminal ops. The second input
// inserts one private op before both yielded values. Explicit source routing
// couples the private branch choice across both yielded paths.

// SYNTH: remark: {{.*}}synth-stat group=multi_yield_shared_superset strategy=mcs reason=success
// SYNTH-SAME: covered=2/2
// SYNTH: fabric.fu
// SYNTH: fabric.op [@arith.addi]
// SYNTH-COUNT-2: fabric.demux
// SYNTH: fabric.op [@arith.xori]
// SYNTH: fabric.mux
// SYNTH: fabric.op [@arith.muli]
// SYNTH: fabric.mux
// SYNTH: fabric.op [@arith.subi]
// SYNTH: fabric.yield

// COUNT: 2

// ENUM: func.func private @fu0_subgraph_0
// ENUM: arith.addi
// ENUM: arith.muli
// ENUM: arith.subi
// ENUM: func.func private @fu0_subgraph_1
// ENUM: arith.addi
// ENUM: arith.xori
// ENUM: arith.muli
// ENUM: arith.subi
// ENUM-NOT: func.func private @fu0_subgraph_2

func.func @pat_two_yields_direct(%a: i32, %b: i32, %c: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield_shared_superset"} {
  %r0, %r1 = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                                %z = %c : i32) -> (i32, i32) {
    %sum = arith.addi %x, %y : i32
    %out0 = arith.muli %sum, %z : i32
    %out1 = arith.subi %sum, %z : i32
    dataflow.yield %out0, %out1 : i32, i32
  }
  return %r0, %r1 : i32, i32
}

func.func @pat_two_yields_private(%a: i32, %b: i32, %c: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield_shared_superset"} {
  %r0, %r1 = dataflow.subgraph(%x = %a : i32, %y = %b : i32,
                                %z = %c : i32) -> (i32, i32) {
    %sum = arith.addi %x, %y : i32
    %p = arith.xori %sum, %z : i32
    %out0 = arith.muli %p, %z : i32
    %out1 = arith.subi %p, %z : i32
    dataflow.yield %out0, %out1 : i32, i32
  }
  return %r0, %r1 : i32, i32
}
