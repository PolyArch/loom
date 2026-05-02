// RUN: loom %s -loom-generalize-subgraphs-to-fu='config=%p/share_recurse_off.yaml dump-stats=true' 2>&1 | FileCheck %s

// Baseline lock-in for `synth.subgraph_share_recurse`. With the knob
// disabled (default), only the standard tier-B mux/demux candidate is
// produced for the tail-extension diff site. The `arith.subi` tail
// fabric.op carries `op_list = [@arith.subi]` only -- no share-aware
// widening. The cost line locks in the floor that the share-recurse-on
// counterpart's `<=` assertion compares against.

// CHECK: remark: {{.*}}synth-stat group=sr_demo strategy=incremental reason=success
// CHECK-SAME: cost=1.940000e+02
// CHECK: fabric.module @fu_sr_demo
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu
// CHECK: fabric.op [@arith.addi]
// CHECK: fabric.demux
// CHECK: fabric.op [@arith.subi]
// CHECK-NOT: fabric.op [@arith.addi, @arith.subi]
// CHECK: fabric.mux
// CHECK: fabric.yield

func.func @sr_pat_addi(%a: i32, %b: i32) -> i32
    attributes {loom.synth_group = "sr_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    dataflow.yield %t : i32
  }
  return %r : i32
}

func.func @sr_pat_addi_then_subi(%a: i32, %b: i32, %c: i32) -> i32
    attributes {loom.synth_group = "sr_demo"} {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %t = arith.addi %x, %y : i32
    %u = arith.subi %t, %z : i32
    dataflow.yield %u : i32
  }
  return %r : i32
}
