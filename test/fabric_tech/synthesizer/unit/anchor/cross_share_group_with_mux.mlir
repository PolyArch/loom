// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/anchor_with_mux.yaml dump-stats=true' 2>&1 | FileCheck %s

// Cross-share-group synthesis routes each shared input through an explicit
// demux and joins branch results with an output mux. Each legal route is a
// complete encoding, and coverage confirms both input functions are matched.

// CHECK: remark: {{.*}}synth-stat group=alu_int_32_x strategy=anchor reason=success
// CHECK-SAME: covered=2/2 nodes=2/1/2 encodings=2
// CHECK: fabric.module @fu_alu_int_32_x
// CHECK-SAME: loom.synthesized_for = "alu_int_32_x"
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK-COUNT-2: fabric.demux
// CHECK-DAG: fabric.op [@arith.addi]
// CHECK-DAG: fabric.op [@arith.muli]
// CHECK: fabric.mux

func.func @pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32_x"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

func.func @pat_muli(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "alu_int_32_x"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.muli %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
