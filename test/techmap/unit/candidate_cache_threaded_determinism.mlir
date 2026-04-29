// RUN: loom-candidate-dump --cache-threads=1 %s > %t.t1
// RUN: loom-candidate-dump --cache-threads=4 %s > %t.t4
// RUN: diff %t.t1 %t.t4
// RUN: FileCheck %s < %t.t1

// The CandidateCache must produce byte-identical output regardless of
// worker-thread count. Two FUs offering arith.addi and arith.muli; a
// chain of ten arith ops of mixed kinds in the graph body. The slot
// indices and template id list ordering must match between threads=1
// and threads=4.

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


fabric.module @fu_muli(%cast0_fu_muli : !fabric.bits<32>, %cast1_fu_muli : !fabric.bits<32>) {
  fabric.spatial_pe(%a = %cast0_fu_muli : !fabric.bits<32>, %b = %cast1_fu_muli : !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  }
  fabric.yield
}


// CHECK: graph #0 @graph_chain
// CHECK-NEXT: op#0 name=arith.addi templates=0
// CHECK-NEXT: op#1 name=arith.muli templates=1
// CHECK-NEXT: op#2 name=arith.addi templates=0
// CHECK-NEXT: op#3 name=arith.muli templates=1
// CHECK-NEXT: op#4 name=arith.addi templates=0
// CHECK-NEXT: op#5 name=arith.muli templates=1
// CHECK-NEXT: op#6 name=arith.addi templates=0
// CHECK-NEXT: op#7 name=arith.muli templates=1
// CHECK-NEXT: op#8 name=arith.addi templates=0
// CHECK-NEXT: op#9 name=arith.muli templates=1
func.func @graph_chain(%a: i32, %b: i32) -> i32 {
  %r = dataflow.graph(%x = %a : i32, %y = %b : i32) -> i32 {
    %v0 = arith.addi %x, %y : i32
    %v1 = arith.muli %v0, %y : i32
    %v2 = arith.addi %v1, %y : i32
    %v3 = arith.muli %v2, %y : i32
    %v4 = arith.addi %v3, %y : i32
    %v5 = arith.muli %v4, %y : i32
    %v6 = arith.addi %v5, %y : i32
    %v7 = arith.muli %v6, %y : i32
    %v8 = arith.addi %v7, %y : i32
    %v9 = arith.muli %v8, %y : i32
    dataflow.yield %v9 : i32
  }
  return %r : i32
}
