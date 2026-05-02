// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_cap_one.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=resource_exhausted

// A single raw tuple candidate is not enough here. The second input starts
// with a same-width, same-op noise node, so a naive first compatible mapping
// pairs the shared chain against the wrong add nodes and fails coverage. MCS
// must keep searching graph-native MCES mappings and return the one verified
// candidate admitted by candidate_cap=1.

// CHECK: remark: {{.*}}synth-stat group=exact_candidate_order_cap_one strategy=mcs reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_exact_candidate_order_cap_one
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi]
// CHECK: fabric.op [@arith.addi]
// CHECK: fabric.demux
// CHECK: fabric.op [@arith.muli]
// CHECK: fabric.mux
// CHECK: fabric.yield

func.func @pat_chain(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "exact_candidate_order_cap_one"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %s0 = arith.addi %x, %y : i32
    %s1 = arith.addi %s0, %z : i32
    dataflow.yield %s1 : i32
  }
  return %r : i32
}

func.func @pat_noisy_chain(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "exact_candidate_order_cap_one"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %n = arith.addi %y, %z : i32
    %s0 = arith.addi %x, %y : i32
    %s1 = arith.addi %s0, %z : i32
    %out = arith.muli %s1, %n : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
