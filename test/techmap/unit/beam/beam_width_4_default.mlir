// Beam width meaningfully changes the partition: width 1 collapses to a
// strict greedy walk and locks in the locally cheapest decision at each
// op, while width 4 keeps the four cheapest partial states and can
// recover when the local optimum forecloses a downstream higher-savings
// fusion.
//
// Adversarial graph: a 4-op linear chain `addi -> muli -> addi -> muli`
// with one extra `arith.muli` reading the second op's result and feeding
// the yield. Templates available are single-op `arith.addi`, single-op
// `arith.muli`, the 2-op pattern `arith.addi -> arith.muli`, and the
// 4-op pattern `addi -> muli -> addi -> muli`. The 4-op pattern would
// match the chain end-to-end, but the side branch reading the second
// muli's input creates a cross edge that makes the locally cheapest
// choice at the chain-tail op leave the addi-muli pair unfused.
//
// The test asserts beam_width=1 and beam_width=4 produce textually
// different IR for the same input — i.e., the two diff invocations both
// fail with non-zero exit, which is what `not diff` is asserting.

// RUN: echo "techmap:" > %t.cfg1.yaml
// RUN: echo "  algorithm: beam" >> %t.cfg1.yaml
// RUN: echo "  beam_width: 1" >> %t.cfg1.yaml
// RUN: echo "techmap:" > %t.cfg4.yaml
// RUN: echo "  algorithm: beam" >> %t.cfg4.yaml
// RUN: echo "  beam_width: 4" >> %t.cfg4.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg1.yaml" > %t.beam1.mlir
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg4.yaml" > %t.beam4.mlir
// RUN: not diff %t.beam1.mlir %t.beam4.mlir
// RUN: FileCheck --check-prefix=BEAM1 %s < %t.beam1.mlir
// RUN: FileCheck --check-prefix=BEAM4 %s < %t.beam4.mlir

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


fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
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


fabric.module @fu_am(%cast0_fu_am : !fabric.bits<32>, %cast1_fu_am : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_am : !fabric.bits<32>, %b = %cast1_fu_am : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %m = fabric.op [@arith.muli] (%k, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %m : !fabric.bits<32>
  }
  }
  fabric.yield
}


fabric.module @fu_amam(%cast0_fu_amam : !fabric.bits<32>, %cast1_fu_amam : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_amam : !fabric.bits<32>, %b = %cast1_fu_amam : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %p = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %q = fabric.op [@arith.muli] (%p, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %r2 = fabric.op [@arith.addi] (%q, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %s = fabric.op [@arith.muli] (%r2, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %s : !fabric.bits<32>
  }
  }
  fabric.yield
}


// BEAM1-LABEL: @graph
// Width 1 commits to a single-op cover at the chain-tail muli (locally
// cheaper because it avoids paying the cross edge that fusion would
// introduce). The result is four subgraphs in body order: the (addi,muli)
// fusion at positions 0..1, the singleton addi at position 2, the
// singleton muli at position 3, and the singleton side-muli at
// position 4.
// BEAM1: dataflow.subgraph
// BEAM1: arith.addi
// BEAM1: arith.muli
// BEAM1: dataflow.yield
// BEAM1: dataflow.subgraph
// BEAM1: arith.addi
// BEAM1: dataflow.yield
// BEAM1: dataflow.subgraph
// BEAM1: arith.muli
// BEAM1: dataflow.yield
// BEAM1: dataflow.subgraph
// BEAM1: arith.muli
// BEAM1: dataflow.yield

// BEAM4-LABEL: @graph
// Width 4 keeps the (addi,muli) cover state alive at the chain-tail and
// emits three subgraphs: the head (addi,muli) at positions 0..1, the
// tail (addi,muli) at positions 2..3, and the side-muli singleton at
// position 4.
// BEAM4: dataflow.subgraph
// BEAM4: arith.addi
// BEAM4: arith.muli
// BEAM4: dataflow.yield
// BEAM4: dataflow.subgraph
// BEAM4: arith.addi
// BEAM4: arith.muli
// BEAM4: dataflow.yield
// BEAM4: dataflow.subgraph
// BEAM4: arith.muli
// BEAM4: dataflow.yield
// BEAM4-NOT: dataflow.subgraph

func.func @graph(%a: i32, %b: i32) -> (i32, i32) {
  %r:2 = dataflow.graph(%x = %a : i32, %y = %b : i32) -> (i32, i32) {
    %a1 = arith.addi %x, %y : i32
    %m2 = arith.muli %a1, %y : i32
    %a3 = arith.addi %m2, %y : i32
    %m4 = arith.muli %a3, %y : i32
    %m_aux = arith.muli %m2, %y : i32
    dataflow.yield %m4, %m_aux : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}
