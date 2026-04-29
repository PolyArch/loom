// RUN: echo "techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" 2> %t.diag | FileCheck %s
// RUN: not test -s %t.diag

// Stress: a complex graph where the FU library contains a multi-op
// template (a muli->addi 2-op chain) plus single-op templates for
// arith.{addi, cmpi}. The ILP partitioner now models multi-op
// coverage natively in the MIP, so no fallback diagnostic is emitted.
// The optimum picks two muli+addi 2-op subgraphs (each fusing a
// (muli, addi) pair) plus a singleton cmpi subgraph, dominating the
// greedy solution that left the second muli at graph level.

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

// CHECK-LABEL: @fu_muli_addi
fabric.module @fu_muli_addi(%cast0_fu_muli_addi : !fabric.bits<32>, %cast1_fu_muli_addi : !fabric.bits<32>) {
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

// CHECK-LABEL: @fu_cmpi
fabric.module @fu_cmpi(%cast0_fu_cmpi : !fabric.bits<1>, %cast1_fu_cmpi : !fabric.bits<1>) {
  fabric.spatial_pe(%a = %cast0_fu_cmpi : !fabric.bits<1>, %b = %cast1_fu_cmpi : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%x = %a : !fabric.bits<1>, %y = %b : !fabric.bits<1>)
                  -> !fabric.bits<1> {
      %k = fabric.op [@arith.cmpi] (%x, %y)
           {hw_params = [{predicate = ["eq"]}]}
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}


// CHECK-LABEL: @graph_chain
// First muli+addi fused into one subgraph (multi-op template wins).
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// Second muli+addi fused into another 2-op subgraph.
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// cmpi gets its own subgraph.
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.cmpi eq
// CHECK-NEXT: dataflow.yield
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_chain(%a: i32, %b: i32) -> (i32, i1) {
  %r:2 = dataflow.graph(%x = %a : i32, %y = %b : i32) -> (i32, i1) {
    %t0 = arith.muli %x, %y : i32
    %t1 = arith.addi %t0, %y : i32
    %t2 = arith.muli %t1, %y : i32
    %t3 = arith.addi %t2, %y : i32
    %p  = arith.cmpi eq, %t1, %t3 : i32
    dataflow.yield %t3, %p : i32, i1
  }
  return %r#0, %r#1 : i32, i1
}
