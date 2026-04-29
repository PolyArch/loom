// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Stress: a graph that mixes a feed-forward fork/join, a carry-driven
// cycle, and a graph-only op (ub.poison) that no FU can absorb. The
// partitioner must:
//   * carve fork/join arith chunks into subgraphs,
//   * keep the carry self-feedback safe (no cross-block SSA cycle),
//   * leave the unsupported ub.poison at graph level.

// CHECK-LABEL: @fu_addsub
fabric.module @fu_addsub(%cast0_fu_addsub : !fabric.bits<32>, %cast1_fu_addsub : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_addsub : !fabric.bits<32>, %b = %cast1_fu_addsub : !fabric.bits<32>) -> !fabric.bits<32> {
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


// CHECK-LABEL: @fu_carry
fabric.module @fu_carry(%c : !fabric.bits<1>, %i : !fabric.bits<1>, %k : !fabric.bits<1>) {
  fabric.spatial_pe(%pc = %c : !fabric.bits<1>,
                    %pi = %i : !fabric.bits<1>,
                    %pk = %k : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%cc = %pc : !fabric.bits<1>,
              %ii = %pi : !fabric.bits<1>,
              %kk = %pk : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@dataflow.carry] (%cc, %ii, %kk)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
             -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @graph_stress
// CHECK: dataflow.graph
// CHECK-DAG: dataflow.subgraph
// ub.poison must remain at graph level (no FU covers it).
// CHECK-DAG: ub.poison
// CHECK: dataflow.yield
func.func @graph_stress(%cond: i1, %init: i32, %a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%c = %cond : i1, %ii = %init : i32,
                      %x = %a : i32, %y = %b : i32) -> i32 {
    %acc = dataflow.carry %c, %ii, %nextcarry : i32
    %sum = arith.addi %x, %y : i32
    %prod = arith.muli %sum, %acc : i32
    %nextcarry = arith.subi %prod, %x : i32
    %junk = ub.poison : i32
    %final = arith.addi %prod, %junk : i32
    dataflow.yield %final : i32
  }
  return %r : i32
}
