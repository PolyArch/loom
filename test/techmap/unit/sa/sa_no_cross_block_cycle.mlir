// RUN: echo "techmap:" > %t.cfg.yaml
// RUN: echo "  algorithm: sa" >> %t.cfg.yaml
// RUN: echo "  sa_steps: 500" >> %t.cfg.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg.yaml" | FileCheck %s

// Adversarial input: a 4-op feedback chain (addi -> muli -> addi -> muli ->
// addi) where naively wrapping each op in its own dataflow.subgraph would
// create a multi-block SSA cycle. The SA partitioner must respect the
// inter-block acyclicity invariant just like greedy does: at least one op
// is left at graph level so the remaining three subgraphs form a DAG.

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


// CHECK-LABEL: @fu_muli
fabric.module @fu_muli {
  %cast0_fu_muli = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %cast1_fu_muli = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_feedback
// CHECK: dataflow.graph
// CHECK: arith.addi
// CHECK: dataflow.subgraph
// CHECK: arith.muli
// CHECK: dataflow.subgraph
// CHECK: arith.addi
// CHECK: dataflow.subgraph
// CHECK: arith.muli
// Only three subgraphs total (the next match is the graph terminator).
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_feedback(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v0 = arith.addi %x, %v3 : i32
    %v1 = arith.muli %v0, %y : i32
    %v2 = arith.addi %v1, %y : i32
    %v3 = arith.muli %v2, %y : i32
    dataflow.yield %v3 : i32
  }
  return %r : i32
}
