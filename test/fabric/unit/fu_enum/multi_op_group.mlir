// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with two configurable fabric.ops, each with multiple op_list members.
// 2 x 3 = 6 supported subgraphs.

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

  // CHECK-DAG: "op#0{op_sel=arith.addi}; op#1{op_sel=arith.andi}"
  // CHECK-DAG: "op#0{op_sel=arith.subi}; op#1{op_sel=arith.andi}"
  // CHECK-DAG: "op#0{op_sel=arith.addi}; op#1{op_sel=arith.ori}"
  // CHECK-DAG: "op#0{op_sel=arith.subi}; op#1{op_sel=arith.ori}"
  // CHECK-DAG: "op#0{op_sel=arith.addi}; op#1{op_sel=arith.xori}"
  // CHECK-DAG: "op#0{op_sel=arith.subi}; op#1{op_sel=arith.xori}"

  return
}
