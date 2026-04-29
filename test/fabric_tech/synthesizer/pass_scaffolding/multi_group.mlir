// RUN: loom %s -loom-generalize-subgraphs-to-fu='dump-stats=true' 2>&1 | FileCheck %s

// Three func.funcs in two groups (`alu` x 2, `fpu` x 1). With
// `dump-stats=true` we expect one `synth-stat` remark per group, ordered
// lexically by group name (`alu` before `fpu`). Both groups fail via the
// stub strategy, so cost/coverage stay at zero.

// CHECK: remark: {{.*}}synth-stat group=alu strategy=incremental_random reason=topology_mismatch cost=0 covered=0/2 nodes=0/0/0
// CHECK: remark: {{.*}}synth-stat group=fpu strategy=incremental_random reason=topology_mismatch cost=0 covered=0/1 nodes=0/0/0

func.func @pat_alu_addi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "alu"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
func.func @pat_alu_subi(%a: i32, %b: i32) -> i32 attributes {loom.synth_group = "alu"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.subi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
func.func @pat_fpu_addf(%a: f32, %b: f32) -> f32 attributes {loom.synth_group = "fpu"} {
  %r = dataflow.subgraph(%x = %a : f32, %y = %b : f32) -> f32 {
    %s = arith.addf %x, %y : f32
    dataflow.yield %s : f32
  }
  return %r : f32
}
