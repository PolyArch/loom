// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with a variadic dataflow.sync of M=3 hardware ports and no
// hw_params (all 2^3-1=7 bitmasks are legal). The enumerator iterates
// every non-zero bitmask, materializes one dataflow.sync per active
// subset, and the isomorphism dedup pass collapses textually-distinct
// bitmasks of the same popcount into a single template (since pairwise
// (input #i, output #i) sync subgraphs of the same N are isomorphic up
// to block-arg permutation). Net: 3 templates remain (N=1, N=2, N=3).

// CHECK-LABEL: fabric.module @fu_sync3
fabric.module @fu_sync3(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
      %u, %v, %w = fabric.op [@dataflow.sync] (%x, %y, %z)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                     -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %u, %v, %w : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

// One-input sync (popcount 1).
// CHECK: func.func private @fu0_subgraph_0
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=100}
// CHECK: dataflow.sync %{{.*}} : (i32) -> i32

// Two-input sync (popcount 2).
// CHECK: func.func private @fu0_subgraph_1
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=110}
// CHECK: dataflow.sync %{{.*}}, %{{.*}} : (i32, i32) -> (i32, i32)

// Three-input sync (popcount 3, the all-ones config).
// CHECK: func.func private @fu0_subgraph_2
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=111}
// CHECK: dataflow.sync %{{.*}}, %{{.*}}, %{{.*}} : (i32, i32, i32) -> (i32, i32, i32)

// No fourth template (extra bitmasks dedup against the above).
// CHECK-NOT: func.func private @fu0_subgraph_3
