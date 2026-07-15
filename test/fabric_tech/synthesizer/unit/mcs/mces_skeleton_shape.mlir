// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/mcs.yaml dump-stats=true' 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=unrealized_conversion_cast

// The MCES skeleton is the shared `arith.addi`. The three tail ops are
// input-specific branches behind the skeleton, so the FU should contain
// one shared add and all three tail operators behind mux/demux hardware.

// CHECK: remark: {{.*}}synth-stat group=mces_three_tail strategy=mcs reason=success
// CHECK-SAME: cost=4.435000e+02
// CHECK-SAME: covered=3/3
// CHECK-SAME: nodes=4/1/2
// CHECK: fabric.module @fu_mces_three_tail
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi]
// CHECK-COUNT-2: fabric.demux
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK-DAG: fabric.op [@arith.divsi]
// CHECK-DAG: fabric.op [@arith.shli]
// CHECK-DAG: fabric.mux
// CHECK: fabric.yield

func.func @pat_add_then_mul(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "mces_three_tail"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.muli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_add_then_div(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "mces_three_tail"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.divsi %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}

func.func @pat_add_then_shl(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "mces_three_tail"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    %out = arith.shli %sum, %z : i32
    dataflow.yield %out : i32
  }
  return %r : i32
}
