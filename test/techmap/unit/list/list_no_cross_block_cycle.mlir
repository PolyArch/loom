// RUN: echo "techmap:" > %t.cfg.yaml
// RUN: echo "  algorithm: list" >> %t.cfg.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg.yaml" | FileCheck %s

// Adversarial input: a 4-op feedback chain (addi -> muli -> addi -> muli ->
// addi) where naively wrapping each op in its own dataflow.subgraph would
// create a multi-block SSA cycle. The list partitioner must detect the
// would-be cycle and leave at least one op at graph level. We verify by
// counting the resulting subgraphs: exactly three of the four ops should
// be wrapped, the fourth (the cycle-closing op) is left as a plain arith
// op inside dataflow.graph. Same outcome as the greedy variant.

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
// CHECK: dataflow.subgraph
// CHECK: arith.addi
// CHECK: dataflow.subgraph
// CHECK: arith.muli
// CHECK: dataflow.subgraph
// CHECK: arith.addi
// Only three subgraphs total. The cycle-closing muli stays at graph level
// as a plain arith op.
// CHECK-NOT: dataflow.subgraph
// CHECK: arith.muli
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
