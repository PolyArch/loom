// RUN: echo "techmap:" > %t.cfg.yaml
// RUN: echo "  algorithm: list" >> %t.cfg.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.cfg.yaml" | FileCheck %s

// Two competing FUs in the library:
//   * @fu_addi: a single-op arith.addi.
//   * @fu_muli_addi: a 2-op chain (arith.muli, arith.addi) with addi as
//     the root.
//
// The graph body is `muli -> addi`. Priority for the root addi op is
//   max_template_size_for_arith.addi * 100 - fanout(addi)
//   = 2 * 100 - 0 = 200
// while priority for the muli is
//   max_template_size_for_arith.muli * 100 - fanout(muli)
//   = 1 * 100 - 1 = 99
// The list scheduler dequeues the addi root first. At that point both the
// single-op addi template and the 2-op (muli, addi) template are
// admissible. The cost model and tie-break (larger size wins) must pick
// the 2-op template, fusing both ops into one subgraph.

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

// CHECK-LABEL: @graph_two_op
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// Only one subgraph: the 2-op fusion wins over the single-op fallback.
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_two_op(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}
