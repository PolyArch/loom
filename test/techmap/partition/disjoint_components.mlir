// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: when the user graph has two disjoint chains (no SSA edges between
// them), the partitioner must produce two independent dataflow.subgraphs
// that share no operands. Both yields come out of distinct subgraph
// results.

// CHECK-LABEL: @fu_addi
fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_disjoint
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK: dataflow.yield
func.func @graph_disjoint(%a: i32, %b: i32, %c: i32, %d: i32) -> (i32, i32) {
  %x, %y = dataflow.graph(%aa = %a : i32, %bb = %b : i32,
                          %cc = %c : i32, %dd = %d : i32) -> (i32, i32) {
    %s1 = arith.addi %aa, %bb : i32
    %s2 = arith.addi %cc, %dd : i32
    dataflow.yield %s1, %s2 : i32, i32
  }
  return %x, %y : i32, i32
}
