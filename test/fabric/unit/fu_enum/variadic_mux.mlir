// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with a variadic dataflow.mux of M=4 hardware data ports plus the
// hardware sel port. hw_params restricts the bitmask iteration to
// {"1100", "0011", "1111"}. The materialized dataflow.mux's sel type
// follows the dataflow.mux verifier: i1 for N==2, `index` for N>=3.
//
// "1100" (N=2) and "0011" (N=2) materialize as isomorphic two-input
// muxes (block-arg permutation only); the dedup pass keeps the first
// occurrence. "1111" (N=4) is structurally distinct from any N==2
// template. Net: 2 templates remain.

// CHECK-LABEL: fabric.module @fu_mux4
fabric.module @fu_mux4 {
  %sel = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %c = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %d = builtin.unrealized_conversion_cast to !fabric.bits<32>
  fabric.spatial_pe(%psel = %sel : !fabric.bits<32>,
                    %pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>,
                    %pd = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%s = %psel : !fabric.bits<32>,
              %x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>, %w = %pd : !fabric.bits<32>)
             -> !fabric.bits<32> {
      %o = fabric.op [@dataflow.mux] (%s, %x, %y, %z, %w)
           {hw_params = [{bitmask = ["1100", "0011", "1111"]}]}
           : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
              !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %o : !fabric.bits<32>
    }
  }
  fabric.yield
}

// First template: N=2 (sel becomes i1).
// CHECK: func.func private @fu0_subgraph_0
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=1100}
// CHECK: dataflow.mux %{{.*}}, %{{.*}}, %{{.*}} : (i1, i32, i32) -> i32

// Second template: N=4 (sel becomes index).
// CHECK: func.func private @fu0_subgraph_1
// CHECK: dataflow.subgraph
// CHECK-SAME: op#0{bitmask=1111}
// CHECK: dataflow.mux %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, i32, i32, i32, i32) -> i32

// "0011" deduped against "1100" (both produce isomorphic 2-input
// muxes). No third template emitted.
// CHECK-NOT: func.func private @fu0_subgraph_2
