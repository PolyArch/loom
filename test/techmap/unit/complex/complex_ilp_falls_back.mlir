// RUN: echo "techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" 2> %t.diag | FileCheck %s
// RUN: FileCheck --check-prefix=DIAG %s < %t.diag

// Stress: a complex graph where the FU library contains a multi-op
// template (a muli->addi 2-op chain) plus single-op templates for
// arith.{addi, cmpi}. The simplified single-op MIP cannot model the
// multi-op coverage so the ILP partitioner emits a diagnostic and
// hands off to greedy. Greedy must produce a structurally valid
// partition: the first muli+addi pair is fused via the 2-op template,
// the second muli is left at graph level (its only template is the
// 2-op chain which would create an inter-block cycle through the
// cmpi consumer), and the trailing addi + cmpi each get their own
// singleton subgraph.

// DIAG: warning: loom-ilp-partitioner: multi-op template candidate detected
// DIAG-SAME: falling back to greedy partitioner

// CHECK-LABEL: @fu_addi
func.func @fu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return
}
// CHECK-LABEL: @fu_muli_addi
func.func @fu_muli_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %m = fabric.op [@arith.addi] (%k, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %m : !fabric.bits<32>
  }
  return
}
// CHECK-LABEL: @fu_cmpi
func.func @fu_cmpi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<1> {
    %k = fabric.op [@arith.cmpi] (%x, %y)
         {hw_params = [{predicate = ["eq"]}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
    fabric.yield %k : !fabric.bits<1>
  }
  return
}

// CHECK-LABEL: @graph_chain
// First muli+addi fused into one subgraph (multi-op template wins).
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// Second muli stays at graph level (cycle break).
// CHECK: arith.muli
// Trailing addi gets a singleton subgraph.
// CHECK: dataflow.subgraph
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
