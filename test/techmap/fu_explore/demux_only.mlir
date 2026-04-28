// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: a single fabric.demux axis. The FU produces an addi result then
// fans it out via a 2-output demux to the FU's two outputs. Each
// (sel, discard, disconnect) choice that yields a non-empty live-yield
// set should produce a template; isomorphic duplicates are deduped.

// CHECK-LABEL: @fu_addi_demux2
func.func @fu_addi_demux2(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %x, %y = fabric.fu(%aa = %a : !fabric.bits<32>,
                     %bb = %b : !fabric.bits<32>)
                    -> (!fabric.bits<32>, !fabric.bits<32>) {
    %k = fabric.op [@arith.addi] (%aa, %bb)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %o0, %o1 = fabric.demux %k : !fabric.bits<32> -> 2
    fabric.yield %o0, %o1 : !fabric.bits<32>, !fabric.bits<32>
  }
  // CHECK: dataflow.subgraph
  // CHECK: arith.addi
  return
}
