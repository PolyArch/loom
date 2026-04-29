// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// A dataflow.graph with one ub.poison (legal at graph level but not
// permitted inside a dataflow.subgraph) and one arith.addi. Greedy must
// wrap the addi in a subgraph and leave the poison op at graph level.

// CHECK-LABEL: @fu_addi
fabric.module @fu_addi {
  %cast0_fu_addi = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %cast1_fu_addi = builtin.unrealized_conversion_cast to !fabric.bits<32>
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


// CHECK-LABEL: @graph_mixed
// CHECK: dataflow.graph
// The poison op stays in the graph body, outside any subgraph.
// CHECK-NEXT: ub.poison
// The addi is wrapped in a subgraph.
// CHECK-NEXT: dataflow.subgraph
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
func.func @graph_mixed(%a: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32) -> i32 {
    %p = ub.poison : i32
    %q = arith.addi %x, %p : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
