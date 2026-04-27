// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU whose only configurable point is a 3-input fabric.mux selecting one of
// three FU inputs to feed an op. 3 supported subgraphs.

// CHECK-LABEL: @fu_mux3_then_op
func.func @fu_mux3_then_op(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                            %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  %r = fabric.fu(%w = %a : !fabric.bits<32>,
                 %x = %b : !fabric.bits<32>,
                 %y = %c : !fabric.bits<32>,
                 %z = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    %sel = fabric.mux %w, %x, %y : !fabric.bits<32>
    %k = fabric.op [@arith.muli] (%sel, %z)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }

  // The three sel values feed three different FU input ports into the
  // same downstream arith.muli. Block-arg permutation is a structural
  // isomorphism, so the three configurations produce graph-isomorphic
  // subgraphs and the enumerator's dedup keeps only the lex-smallest one
  // (sel=0).
  // CHECK: mux#0{sel=0,discard=false,disconnect=false}
  // CHECK-NOT: mux#0{sel=1,discard=false,disconnect=false}
  // CHECK-NOT: mux#0{sel=2,discard=false,disconnect=false}

  return
}
