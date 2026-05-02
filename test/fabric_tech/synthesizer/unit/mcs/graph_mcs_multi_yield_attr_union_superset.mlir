// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYNTH --implicit-check-not=unrealized_conversion_cast
// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost.yaml --print-stats=false --print-wallclock=false %s \
// RUN:   | loom - -loom-enumerate-fu-subgraphs > %t.enum.mlir
// RUN: grep -c "func.func private @fu0_subgraph_" %t.enum.mlir \
// RUN:   | FileCheck %s --check-prefix=COUNT
// RUN: FileCheck %s --check-prefix=ENUM < %t.enum.mlir

// The shared stream has independent attribute choices while the first
// yielded value has an arithmetic choice. Enumeration exposes the strict
// superset from those independent choices and the optional output routes,
// while the second yield carries the shared stream index.

// SYNTH: remark: {{.*}}synth-stat group=multi_yield_attr_union strategy=mcs reason=success
// SYNTH-SAME: covered=2/2
// SYNTH: fabric.fu
// SYNTH: fabric.op [@dataflow.stream]
// SYNTH-SAME: cont_cond = ["<", ">"]
// SYNTH-SAME: step_op = ["+=", "-="]
// SYNTH: fabric.demux
// SYNTH-DAG: fabric.op [@arith.addi]
// SYNTH-DAG: fabric.op [@arith.muli]
// SYNTH: fabric.mux
// SYNTH: fabric.yield

// COUNT: 20

// ENUM: func.func private @fu0_subgraph_0
// ENUM: dataflow.stream
// ENUM: arith.addi
// ENUM: func.func private @fu0_subgraph_8
// ENUM: dataflow.stream
// ENUM: arith.muli
// ENUM: func.func private @fu0_subgraph_19
// ENUM: dataflow.stream
// ENUM: arith.muli
// ENUM-NOT: func.func private @fu0_subgraph_20

func.func @pat_stream_add_inc(%lb: i32, %ub: i32, %step: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield_attr_union"} {
  %value, %idx_out = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                                       %s = %step : i32) -> (i32, i32) {
    %idx, %rwc = dataflow.stream %l, %u, %s
                {step_op = "+=", cont_cond = "<"} : i32
    %out = arith.addi %idx, %s : i32
    dataflow.yield %out, %idx : i32, i32
  }
  return %value, %idx_out : i32, i32
}

func.func @pat_stream_mul_dec(%lb: i32, %ub: i32, %step: i32) -> (i32, i32)
    attributes {loom.synth_group = "multi_yield_attr_union"} {
  %value, %idx_out = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                                       %s = %step : i32) -> (i32, i32) {
    %idx, %rwc = dataflow.stream %l, %u, %s
                {step_op = "-=", cont_cond = ">"} : i32
    %out = arith.muli %idx, %s : i32
    dataflow.yield %out, %idx : i32, i32
  }
  return %value, %idx_out : i32, i32
}
