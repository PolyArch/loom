// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// User Example 1: an FU with a single fabric.op offering {arith.addi, arith.subi}.
// The enumerator should produce two dataflow.subgraph candidates, one wrapping
// arith.addi, one wrapping arith.subi.

// CHECK-LABEL: fabric.module @fu_addi_or_subi
fabric.module @fu_addi_or_subi(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi, @arith.subi] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK: fabric.fu

// CHECK: dataflow.subgraph
// CHECK-SAME: loom.from_fu_config = "op#0{op_sel=arith.addi}"
// CHECK:   arith.addi
// CHECK:   dataflow.yield

// CHECK: dataflow.subgraph
// CHECK-SAME: loom.from_fu_config = "op#0{op_sel=arith.subi}"
// CHECK:   arith.subi
// CHECK:   dataflow.yield
