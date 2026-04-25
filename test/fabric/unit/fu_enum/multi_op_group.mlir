// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with two configurable fabric.ops, each with multiple op_list members.
// 2 x 2 = 4 supported subgraphs.

// CHECK-LABEL: @fu_two_op_groups
func.func @fu_two_op_groups(%a: !fabric.bits<16>, %b: !fabric.bits<16>) {
  %r = fabric.fu(%x = %a : !fabric.bits<16>, %y = %b : !fabric.bits<16>)
                -> !fabric.bits<16> {
    %k = fabric.op [@arith.addi, @arith.subi] (%x, %y)
         : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
    %m = fabric.op [@arith.andi, @arith.ori, @arith.xori] (%k, %y)
         : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
    fabric.yield %m : !fabric.bits<16>
  }

  // 2 (add/sub) x 3 (and/or/xor) = 6 enumerated subgraphs.
  // CHECK: dataflow.subgraph
  // CHECK-SAME: "op#0=arith.addi; op#1=arith.andi"
  // CHECK: dataflow.subgraph
  // CHECK-SAME: "op#0=arith.subi; op#1=arith.andi"
  // CHECK: dataflow.subgraph
  // CHECK-SAME: "op#0=arith.addi; op#1=arith.ori"
  // CHECK: dataflow.subgraph
  // CHECK-SAME: "op#0=arith.subi; op#1=arith.ori"
  // CHECK: dataflow.subgraph
  // CHECK-SAME: "op#0=arith.addi; op#1=arith.xori"
  // CHECK: dataflow.subgraph
  // CHECK-SAME: "op#0=arith.subi; op#1=arith.xori"

  return
}
