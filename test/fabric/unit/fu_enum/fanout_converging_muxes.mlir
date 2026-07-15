// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Direct SSA multi-use is broadcast. Selecting one result cannot make the
// other operation disappear because both operations still receive every input
// token. No configuration is valid without explicit input routing.

// CHECK-LABEL: fabric.module @implicit_add_or_mul
fabric.module @implicit_add_or_mul(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %add = fabric.op [@arith.addi] (%x, %y)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %mul = fabric.op [@arith.muli] (%x, %y)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %out = fabric.mux %add, %mul : !fabric.bits<32>
      fabric.yield %out : !fabric.bits<32>
    }
  }
  fabric.yield
}

// Explicit demuxes route each input to exactly one operation. The result mux
// must select the same branch, leaving two valid configured functions.

// CHECK-LABEL: fabric.module @explicit_add_or_mul
fabric.module @explicit_add_or_mul(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %x0, %x1 = fabric.demux %x : !fabric.bits<32> -> 2
      %y0, %y1 = fabric.demux %y : !fabric.bits<32> -> 2
      %add = fabric.op [@arith.addi] (%x0, %y0)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %mul = fabric.op [@arith.muli] (%x1, %y1)
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %out = fabric.mux %add, %mul : !fabric.bits<32>
      fabric.yield %out : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-NOT: func.func private @fu0_subgraph_
// CHECK: func.func private @fu1_subgraph_0
// CHECK: dataflow.subgraph
// CHECK-SAME: demux#0{sel=0,discard=false,disconnect=false}; demux#1{sel=0,discard=false,disconnect=false}; mux#0{sel=0,discard=false,disconnect=false}
// CHECK: arith.addi
// CHECK-NOT: arith.muli

// CHECK: func.func private @fu1_subgraph_1
// CHECK: dataflow.subgraph
// CHECK-SAME: demux#0{sel=1,discard=false,disconnect=false}; demux#1{sel=1,discard=false,disconnect=false}; mux#0{sel=1,discard=false,disconnect=false}
// CHECK: arith.muli
// CHECK-NOT: func.func private @fu1_subgraph_2
