// RUN: loom-candidate-dump %s | FileCheck %s

// Two FUs in the module: an addi-FU (template id 0) and a muli-FU
// (template id 1). The graph mixes addi/muli ops with one ub.poison op
// that is legal at graph-level but not fabric-supported. The candidate
// cache must:
//   * record one entry per non-terminator op in graph body program order;
//   * report sorted, ascending template ids for fabric-supported ops;
//   * report `<empty>` for non-fabric-supported ops.

fabric.module @fu_addi(%cast0_fu_addi : !fabric.bits<32>, %cast1_fu_addi : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_addi : !fabric.bits<32>, %b = %cast1_fu_addi : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK: graph #0 @graph_mixed
func.func @graph_mixed(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    // CHECK-NEXT: op#0 name=arith.addi templates=0
    %s = arith.addi %x, %y : i32
    // CHECK-NEXT: op#1 name=arith.muli templates=1
    %t = arith.muli %s, %y : i32
    // CHECK-NEXT: op#2 name=ub.poison templates=<empty>
    %p = ub.poison : i32
    // CHECK-NEXT: op#3 name=arith.addi templates=0
    %u = arith.addi %t, %p : i32
    dataflow.yield %u : i32
  }
  return %r : i32
}
