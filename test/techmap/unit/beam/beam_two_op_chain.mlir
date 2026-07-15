// RUN: echo "fabric_techmap:" > %t.cfg.yaml
// RUN: echo "  algorithm: beam" >> %t.cfg.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg.yaml" | FileCheck %s

// Same input as greedy_two_op_chain.mlir: an FU implements arith.muli
// followed by arith.addi. The dataflow.graph contains exactly that chain.
// The beam-search partitioner must fuse both ops into a single
// dataflow.subgraph, identical to greedy's behavior. With beam_width
// defaulting to 4 the search trivially picks the densest 2-op cover.

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


// CHECK-LABEL: @graph_two_op
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// Only one subgraph should be emitted. The graph terminator immediately
// follows the single subgraph.
// CHECK: dataflow.yield
func.func @graph_two_op(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
