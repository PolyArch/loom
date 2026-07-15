// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_low_route_cost_cap_one.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=resource_exhausted

// The second input starts with a compatible decoy addi. A bounded DFS that
// stops after the earliest raw tuple spends the whole cap before reaching the
// true shared addi/muli chain. Exact graph-MCES search should still return a
// verified graph-native structure admitted by candidate_cap=1.

// CHECK: remark: {{.*}}synth-stat group=decoy_candidate_order strategy=mcs reason=success
// CHECK-SAME: covered=2/2
// CHECK-SAME: nodes=3/2/4
// CHECK: fabric.module @fu_decoy_candidate_order
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi]
// CHECK: fabric.demux
// CHECK: fabric.op [@arith.addi]
// CHECK: fabric.mux
// CHECK: fabric.mux
// CHECK: fabric.op [@arith.muli]
// CHECK: fabric.yield

func.func @pat_chain(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "decoy_candidate_order"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.muli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_ordered(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "decoy_candidate_order"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %decoy = arith.addi %z, %y : i32
    %sum = arith.addi %x, %y : i32
    %out = arith.muli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
