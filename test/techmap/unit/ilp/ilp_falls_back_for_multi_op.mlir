// When the template library contains a multi-op template, the simplified
// single-op MIP cannot model coverage and the ILP partitioner must fall
// back to greedy, emitting a diagnostic on the way out. The greedy
// fallback fuses the chain into a single subgraph.

// RUN: echo "techmap:" > %t.ilp.yaml
// RUN: echo "  algorithm: ilp" >> %t.ilp.yaml
// RUN: loom %s -loom-partition-graph-into-subgraphs="config=%t.ilp.yaml" 2> %t.diag | FileCheck %s
// RUN: FileCheck --check-prefix=DIAG %s < %t.diag

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

// Greedy fallback: muli+addi fuse into a single subgraph.
// CHECK-LABEL: @graph_two_op
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK: dataflow.yield
func.func @graph_two_op(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %p = arith.muli %x, %y : i32
    %q = arith.addi %p, %y : i32
    dataflow.yield %q : i32
  }
  return %r : i32
}

// DIAG: warning: loom-ilp-partitioner: multi-op template candidate detected
// DIAG-SAME: falling back to greedy partitioner
