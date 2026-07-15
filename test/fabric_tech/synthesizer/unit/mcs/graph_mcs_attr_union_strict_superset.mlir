// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYNTH --implicit-check-not=unrealized_conversion_cast
// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost.yaml --print-stats=false --print-wallclock=false %s \
// RUN:   | loom - -loom-enumerate-fu-subgraphs > %t.enum.mlir
// RUN: grep -c "func.func private @fu0_subgraph_" %t.enum.mlir \
// RUN:   | FileCheck %s --check-prefix=COUNT
// RUN: FileCheck %s --check-prefix=ENUM < %t.enum.mlir

// The shared stream has different software attributes in the two inputs.
// graph-MCS should still share one fabric.op[@dataflow.stream] by lifting
// both observed values into hw_params. The independent stream attribute
// axes and the route arm create a strict superset of the two inputs.

// SYNTH: remark: {{.*}}synth-stat group=attr_union_superset strategy=mcs reason=success
// SYNTH-SAME: covered=2/2
// SYNTH-SAME: nodes=3/1/2
// SYNTH: fabric.fu
// SYNTH: fabric.op [@dataflow.stream]
// SYNTH-SAME: cont_cond = ["<", ">"]
// SYNTH-SAME: step_op = ["+=", "-="]
// SYNTH-COUNT-2: fabric.demux
// SYNTH: fabric.op [@arith.muli]
// SYNTH: fabric.mux
// SYNTH: fabric.op [@arith.addi]

// COUNT: 8

// ENUM: func.func private @fu0_subgraph_0
// ENUM: dataflow.stream
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_1
// ENUM: dataflow.stream
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_2
// ENUM: dataflow.stream
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_3
// ENUM: dataflow.stream
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_4
// ENUM: dataflow.stream
// ENUM: arith.muli
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_5
// ENUM: dataflow.stream
// ENUM: arith.muli
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_6
// ENUM: dataflow.stream
// ENUM: arith.muli
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_7
// ENUM: dataflow.stream
// ENUM: arith.muli
// ENUM: arith.addi
// ENUM-NOT: func.func private @fu0_subgraph_8

func.func @pat_direct_inc(%lb: i32, %ub: i32, %step: i32) -> i32
    attributes {loom.synth_group = "attr_union_superset"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32) -> i32 {
    %i, %rwc = dataflow.stream %l, %u, %s
                {step_op = "+=", cont_cond = "<"} : i32
    %v = arith.addi %i, %s : i32
    dataflow.yield %v : i32
  }
  return %r : i32
}

func.func @pat_private_dec(%lb: i32, %ub: i32, %step: i32) -> i32
    attributes {loom.synth_group = "attr_union_superset"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32) -> i32 {
    %i, %rwc = dataflow.stream %l, %u, %s
                {step_op = "-=", cont_cond = ">"} : i32
    %p = arith.muli %i, %s : i32
    %v = arith.addi %p, %s : i32
    dataflow.yield %v : i32
  }
  return %r : i32
}
