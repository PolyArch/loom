// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: an FU advertising a single fabric.op[@arith.select] has no
// sw_configs axis (arith.select is non-configurable per the design
// statement), so the enumerator emits exactly one template wrapping
// arith.select.
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout (sel is fixed bits<1> and the data ports accept any width).

// CHECK-LABEL: fabric.module @fu_select
fabric.module @fu_select {
  %s = builtin.unrealized_conversion_cast to !fabric.bits<1>
  %a = builtin.unrealized_conversion_cast to !fabric.bits<1>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<1>
  fabric.spatial_pe(%ps = %s : !fabric.bits<1>,
                    %pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%ss = %ps : !fabric.bits<1>,
              %aa = %pa : !fabric.bits<1>,
              %bb = %pb : !fabric.bits<1>) -> !fabric.bits<1> {
      %k = fabric.op [@arith.select] (%ss, %aa, %bb)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK: dataflow.subgraph
// CHECK: arith.select
// CHECK-NOT: dataflow.subgraph
