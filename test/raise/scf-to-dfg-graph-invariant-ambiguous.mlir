// RUN: not loom-raise-opt --loom-lower-graph-invariant %s 2>&1 | FileCheck %s

// CHECK: error: loom-lower-graph-invariant: graph requires invariant lowering but has multiple directly owned dataflow.stream phase owners

dataflow.graph.func private @g_ambiguous_stream_phase(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64,
    %arg5: i64) -> (none, i64) {
  %iv0, %phase0 = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %iv1, %phase1 = dataflow.stream %arg1, %arg4, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %sum = arith.addi %arg5, %arg5 : i64
  dataflow.graph.return %arg0, %sum : none, i64
}
