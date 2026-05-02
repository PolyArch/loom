// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GRAPH --implicit-check-not=unrealized_conversion_cast

// This cyclic pair already has a graph-native MCES candidate. With hidden
// compatibility fallback enabled, MCS can choose the incremental-shaped FU
// instead. Once MCS owns this case directly, the graph-MCES structure should
// be selected without relying on an inner strategy.

// GRAPH: remark: {{.*}}synth-stat group=graph_native_accum strategy=mcs reason=success
// GRAPH-SAME: covered=2/2
// GRAPH-SAME: nodes=4/1/2
// GRAPH: fabric.module @fu_graph_native_accum
// GRAPH: fabric.fu
// GRAPH-DAG: fabric.op [@dataflow.stream]
// GRAPH-DAG: fabric.demux
// GRAPH-DAG: fabric.demux
// GRAPH-DAG: fabric.op [@arith.addi]
// GRAPH-DAG: fabric.op [@arith.xori]
// GRAPH-DAG: fabric.mux
// GRAPH-DAG: fabric.op [@dataflow.carry]
// GRAPH: fabric.yield

func.func @pat_accum_addi(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "graph_native_accum"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.addi %c, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}

func.func @pat_accum_xori(%lb: i32, %ub: i32, %step: i32, %init: i32) -> i32
    attributes {loom.synth_group = "graph_native_accum"} {
  %r = dataflow.subgraph(%l = %lb : i32, %u = %ub : i32,
                         %s = %step : i32, %in = %init : i32) -> i32 {
    %idx, %rwc = dataflow.stream %l, %u, %s
                 {step_op = "+=", cont_cond = "<"} : i32
    %c = dataflow.carry %rwc, %in, %nxt : i32
    %nxt = arith.xori %c, %idx : i32
    dataflow.yield %c : i32
  }
  return %r : i32
}
