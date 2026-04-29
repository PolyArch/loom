// Beam search with beam_width=1 collapses to a strict greedy walk: the
// per-step ranking keeps exactly one state, the locally-cheapest cover.
// We verify that the resulting IR matches the greedy partitioner's output
// for the same input.

// RUN: echo "techmap:" > %t.greedy.yaml
// RUN: echo "  algorithm: greedy" >> %t.greedy.yaml
// RUN: echo "techmap:" > %t.beam1.yaml
// RUN: echo "  algorithm: beam" >> %t.beam1.yaml
// RUN: echo "  beam_width: 1" >> %t.beam1.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.greedy.yaml" > %t.greedy.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.beam1.yaml" > %t.beam1.mlir
// RUN: diff %t.greedy.mlir %t.beam1.mlir

// Two competing FUs in the library:
//   * @fu_addi: a single-op arith.addi.
//   * @fu_muli_addi: a 2-op chain (arith.muli, arith.addi) with addi as
//     the root.
//
// The graph body is `muli -> addi`. The beam-width=1 partitioner walks
// the same reverse-topo order as greedy and must pick the 2-op cover at
// the addi root. The diff above asserts byte-equivalent IR.

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


fabric.module @fu_muli_addi {
  %cast0_fu_muli_addi = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %cast1_fu_muli_addi = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%a = %cast0_fu_muli_addi : !fabric.bits<32>, %b = %cast1_fu_muli_addi : !fabric.bits<32>) -> !fabric.bits<32> {
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


func.func @graph_two_op(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
