// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost.yaml --print-stats=false --print-wallclock=false %s > %t.workers1.mlir
// RUN: loom-synth-fu-dump --config=%p/mcs_low_route_cost_workers_4.yaml --print-stats=false --print-wallclock=false %s > %t.workers4.mlir
// RUN: FileCheck %s --check-prefix=GRAPH --implicit-check-not=unrealized_conversion_cast < %t.workers1.mlir
// RUN: FileCheck %s --check-prefix=GRAPH --implicit-check-not=unrealized_conversion_cast < %t.workers4.mlir
// RUN: diff %t.workers1.mlir %t.workers4.mlir

// `branch_workers` should parallelize or shard graph-MCES work without
// changing the chosen graph-native structure.

// GRAPH: fabric.module @fu_worker_determinism_accum
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
    attributes {loom.synth_group = "worker_determinism_accum"} {
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
    attributes {loom.synth_group = "worker_determinism_accum"} {
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
