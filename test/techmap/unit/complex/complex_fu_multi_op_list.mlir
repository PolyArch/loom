// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: a single FU exposes a multi-element op_list spanning the
// {addi, subi} hardware-share group; a separate single-op @arith.muli FU
// covers the remaining op kind. The graph interleaves all three op kinds
// across five program-ordered nodes. Each op is covered by a singleton
// dataflow.subgraph (the multi-element op_list does not by itself fuse
// distinct graph ops -- it only widens the FU's enumerated template
// catalog so addi and subi share the same hardware tile).
//
// NOTE: the brief originally asked for op_list = [addi, subi, muli] but
// arith.muli is not in the {addi, subi} hardware-share group, so that
// triple is rejected by fabric.op verification. Using the legal
// {addi, subi} group plus a separate muli FU exercises the same intent:
// a multi-element op_list FU is in the library and every graph op kind
// is reachable.

// CHECK-LABEL: @fu_addsub
fabric.module @fu_addsub(%cast0_fu_addsub : !fabric.bits<32>, %cast1_fu_addsub : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_addsub : !fabric.bits<32>, %b = %cast1_fu_addsub : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi, @arith.subi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @fu_muli
fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_mixed
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.subi
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.subi
// Exactly five subgraphs; the next dataflow.* match is the graph terminator.
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_mixed(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.addi %x, %y : i32
    %q = arith.subi %p, %y : i32
    %s = arith.muli %q, %y : i32
    %t = arith.addi %s, %y : i32
    %u = arith.subi %t, %y : i32
    dataflow.yield %u : i32
  }
  return %r : i32
}
