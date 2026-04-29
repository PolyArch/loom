// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU offers {addi, subi}. Three pattern subgraphs: addi (matches), subi
// (matches), muli (no match - muli isn't in the op_list).

fabric.module @hw_addsub(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
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

// CHECK-LABEL: @pat_addi
func.func @pat_addi(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{op_sel=arith.addi}"
  // CHECK-SAME: loom.matched_fu = "@hw_addsub#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.addi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_subi
func.func @pat_subi(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.match_config = "op#0{op_sel=arith.subi}"
  // CHECK-SAME: loom.matched_fu = "@hw_addsub#0"
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.subi %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}

// CHECK-LABEL: @pat_muli_unmatched
func.func @pat_muli_unmatched(%x: i32, %y: i32) -> i32 {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.unmatched
  %r = dataflow.subgraph(%a = %x : i32, %b = %y : i32) -> i32
       attributes {loom.is_pattern} {
    %k = arith.muli %a, %b : i32
    dataflow.yield %k : i32
  }
  return %r : i32
}
