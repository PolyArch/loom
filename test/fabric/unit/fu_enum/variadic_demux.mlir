// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with a variadic dataflow.demux of M=4 hardware data outputs and
// hw_params restricting bitmask iteration to {"1100","0011","1111"}.
// "1100" and "0011" both yield N=2 demuxes that dedup against each
// other; "1111" yields a 4-output demux. The materialized sel type
// follows the dataflow.demux verifier (i1 for N==2, index for N>=3).

// CHECK-LABEL: fabric.module @fu_demux4
fabric.module @fu_demux4(%sel : !fabric.bits<32>, %in : !fabric.bits<32>) {
  fabric.pe [spatial] (%psel = %sel : !fabric.bits<32>,
                    %pin = %in : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>,
                 !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%s = %psel : !fabric.bits<32>,
              %x = %pin : !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>,
                 !fabric.bits<32>, !fabric.bits<32>) {
      %a, %b, %c, %d = fabric.op [@dataflow.demux] (%s, %x)
                       {hw_params = [{bitmask = ["1100", "0011", "1111"]}]}
                       : (!fabric.bits<32>, !fabric.bits<32>)
                         -> (!fabric.bits<32>, !fabric.bits<32>,
                             !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %a, %b, %c, %d : !fabric.bits<32>, !fabric.bits<32>,
                                    !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

// First template: N=2 (sel becomes i1).
// CHECK: func.func private @fu0_subgraph_0
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=1100}
// CHECK: dataflow.demux %{{.*}}, %{{.*}} : (i1, i32) -> (i32, i32)

// Second template: N=4 (sel becomes index).
// CHECK: func.func private @fu0_subgraph_1
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=1111}
// CHECK: dataflow.demux %{{.*}}, %{{.*}} : (index, i32) -> (i32, i32, i32, i32)

// "0011" deduped against "1100".
// CHECK-NOT: func.func private @fu0_subgraph_2
