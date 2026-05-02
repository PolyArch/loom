// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs_cap_one.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=resource_exhausted

// This pure-DAG group has one actual MCES candidate: the shared addi
// skeleton with one incompatible tail per input. A candidate cap of one
// should allow that candidate instead of rejecting planned fallback
// branch orderings before MCES enumeration starts.

// CHECK: remark: {{.*}}synth-stat group=cap_one_mces strategy=mcs reason=success
// CHECK-SAME: covered=2/2
// CHECK: fabric.module @fu_cap_one_mces
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK-DAG: fabric.op [@arith.divsi]
// CHECK: fabric.yield

func.func @pat_add_then_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "cap_one_mces"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.muli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_add_then_div(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "cap_one_mces"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.divsi %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
