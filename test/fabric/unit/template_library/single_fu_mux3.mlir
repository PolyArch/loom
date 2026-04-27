// RUN: loom-template-dump %s | FileCheck %s

// One FU with a 3-input mux feeding a multiplier. TemplateLibrary must
// emit three templates rooted at arith.muli, all of size 1 (the chosen
// input is wired straight into the muli; the mux is consumed by the
// configuration).

func.func @fu_mux3(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
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
  return
}

// All three sel values produce graph-isomorphic single-op templates
// (a 2-input arith.muli over two distinct block args). Enumerator dedup
// keeps only the lex-smallest, so just one template is emitted.
// CHECK: tpl#0 root=arith.muli size=1 cfg=mux#0{sel=0,discard=false,disconnect=false}
// CHECK-NOT: tpl#1
// CHECK-NOT: mux#0{sel=1
// CHECK-NOT: mux#0{sel=2
