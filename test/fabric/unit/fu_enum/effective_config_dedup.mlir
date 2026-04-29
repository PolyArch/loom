// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU implements either a*b or a*b+c via demux/mux. The demux selects
// whether the multiplier output is yielded directly (sel=0) or fed into
// the addi (sel=1); the output mux re-merges the two paths.
//
// Naively this looks like a 2x2 = 4 Cartesian product over the two sel
// knobs, but only 2 of those configurations produce valid effective
// computes (the other 2 leave the yield path with no live source). After
// alive-fires fixed-point culling and isomorphism dedup the enumerator
// emits exactly 2 templates: a*b and a*b+c. This test pins that count.

// CHECK-LABEL: fabric.module @fu_mul_or_mac
fabric.module @fu_mul_or_mac(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %mul = fabric.op [@arith.muli] (%x, %y)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %d0, %d1 = fabric.demux %mul : !fabric.bits<32> -> 2
      %add = fabric.op [@arith.addi] (%d1, %z)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %out = fabric.mux %d0, %add : !fabric.bits<32>
      fabric.yield %out : !fabric.bits<32>
    }
  }
  fabric.yield
}

// Exactly two wrappers emitted: subgraph_0 and subgraph_1, no third.
// CHECK: func.func private @fu0_subgraph_0
// CHECK: dataflow.subgraph
// CHECK-SAME: demux#0{sel=0,discard=false,disconnect=false}; mux#0{sel=0,discard=false,disconnect=false}
// CHECK:   arith.muli
// CHECK:   dataflow.yield

// CHECK: func.func private @fu0_subgraph_1
// CHECK: dataflow.subgraph
// CHECK-SAME: demux#0{sel=1,discard=false,disconnect=false}; mux#0{sel=1,discard=false,disconnect=false}
// CHECK:   arith.muli
// CHECK:   arith.addi
// CHECK:   dataflow.yield

// No third template.
// CHECK-NOT: func.func private @fu0_subgraph_2
