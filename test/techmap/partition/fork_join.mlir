// RUN: loom %s -loom-partition-graph-into-subgraphs | FileCheck %s

// Pins: a multi-op FU implementing a mac (mul-then-add) must match a
// fork/join user-graph shape that has the multiply's result on the
// SECOND operand of the addi (commutativity swap). The partitioner uses
// VF2 isomorphism, so the swap should not break the bind. Exactly one
// dataflow.subgraph wraps both ops.

// CHECK-LABEL: @fu_mac
func.func @fu_mac(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                  %c: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %s = fabric.op [@arith.addi] (%m, %z)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %s : !fabric.bits<32>
  }
  return
}

// CHECK-LABEL: @graph_mac_commuted
// CHECK: dataflow.graph
// CHECK: dataflow.subgraph
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: dataflow.yield
// CHECK-NOT: dataflow.subgraph
// CHECK: dataflow.yield
func.func @graph_mac_commuted(%a: i32, %b: i32, %c: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32, %z = %c : i32) -> i32 {
    %m = arith.muli %x, %y : i32
    // muli's result on the second operand of addi.
    %s = arith.addi %z, %m : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}
