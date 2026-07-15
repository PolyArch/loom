// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// The sync fires with only input #1 active. %sum also feeds the yielded
// multiply, so its masked use at sync input #0 cannot be treated as an
// implicit drain for the broadcast token.

// CHECK-LABEL: fabric.module @masked_variadic_port_broadcast
fabric.module @masked_variadic_port_broadcast(%a : !fabric.bits<32>,
                                               %b : !fabric.bits<32>,
                                               %c : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>,
                       %pc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addi] (%x, %y)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %sync:2 = fabric.op [@dataflow.sync] (%sum, %z)
                {hw_params = [{bitmask = ["01"]}]}
                : (!fabric.bits<32>, !fabric.bits<32>)
                  -> (!fabric.bits<32>, !fabric.bits<32>)
      %out = fabric.op [@arith.muli] (%sum, %sync#1)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %out : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-NOT: func.func private @fu0_subgraph_
