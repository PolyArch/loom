// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins the invariant that analyzeConfig's final validate predicate only
// rejects configs whose firing ops have a dead active operand. A
// non-firing internal op whose own operands fell out of `alive` (because
// its variadic upstream's bitmask masked off the corresponding output)
// must not cause the surrounding config to be discarded.
//
// Shape: a variadic dataflow.demux with M=3 output ports drives three
// distinct downstream consumers (addi/muli/andi). The hw_params bitmask
// allow-set picks two of three outputs at a time, leaving one demux
// output dead and its consumer non-firing. The pre-fix validator
// rejected every such config because the dead consumer's operand was
// "demanded but not alive". After the fix, three templates emerge --
// one per bitmask -- each materializing the two surviving consumers.

// CHECK-LABEL: fabric.module @fu_demux_per_output_consumer
fabric.module @fu_demux_per_output_consumer(%sel : !fabric.bits<32>, %x : !fabric.bits<32>) {
  fabric.spatial_pe(%psel = %sel : !fabric.bits<32>,
                    %px = %x : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%s = %psel : !fabric.bits<32>, %y = %px : !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
      %a, %b, %c = fabric.op [@dataflow.demux] (%s, %y)
                   {hw_params = [{bitmask = ["110", "101", "011"]}]}
                   : (!fabric.bits<32>, !fabric.bits<32>)
                     -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      %t0 = fabric.op [@arith.addi] (%a, %a)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %t1 = fabric.op [@arith.muli] (%b, %b)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %t2 = fabric.op [@arith.andi] (%c, %c)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %t0, %t1, %t2 : !fabric.bits<32>, !fabric.bits<32>,
                                    !fabric.bits<32>
    }
  }
  fabric.yield
}

// Three templates, one per bitmask. Each surviving template materializes
// the two consumers whose demux outputs were active under that bitmask.

// bitmask=110: outputs #0 and #1 active; addi and muli survive.
// CHECK: func.func private @fu0_subgraph_0
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=110}
// CHECK: dataflow.demux
// CHECK: arith.addi
// CHECK: arith.muli

// bitmask=101: outputs #0 and #2 active; addi and andi survive.
// CHECK: func.func private @fu0_subgraph_1
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=101}
// CHECK: dataflow.demux
// CHECK: arith.addi
// CHECK: arith.andi

// bitmask=011: outputs #1 and #2 active; muli and andi survive.
// CHECK: func.func private @fu0_subgraph_2
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=011}
// CHECK: dataflow.demux
// CHECK: arith.muli
// CHECK: arith.andi
