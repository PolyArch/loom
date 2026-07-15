// When the template library contains a multi-op template, the ILP
// partitioner models multi-op coverage natively in the MIP via rooted
// VF2 candidates. The optimum binds the muli+addi chain to a single
// 2-op subgraph; no greedy fallback diagnostic is emitted.

// RUN: echo "fabric_techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" 2> %t.diag | FileCheck %s
// RUN: not test -s %t.diag

// CHECK-LABEL: @fu_muli_addi
fabric.module @fu_muli_addi(%cast0_fu_muli_addi : !fabric.bits<32>, %cast1_fu_muli_addi : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_muli_addi : !fabric.bits<32>, %b = %cast1_fu_muli_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %m = fabric.op [@arith.addi] (%k, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %m : !fabric.bits<32>
  }
  }
  fabric.yield
}


// ILP-native multi-op coverage: muli+addi fuse into a single subgraph.
// CHECK-LABEL: @graph_two_op
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK: dataflow.yield
func.func @graph_two_op(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
