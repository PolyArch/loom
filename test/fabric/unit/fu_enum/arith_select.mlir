// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU implementing arith.select. The op has a fixed (i1 sel, T data,
// T data) -> T schema and no configurable knobs, so the enumerator
// emits exactly one template.
//
// Note: arith.select has strict-SSA eager-evaluation semantics
// (consumes both data inputs regardless of sel). It is distinct from
// dataflow.mux (which has data-dependent gating). Patterns containing
// arith.select must NOT match a dataflow.mux fabric.op flavor and
// vice versa.
//
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout: sel and data are all bits<1>.

// CHECK-LABEL: fabric.module @fu_select
fabric.module @fu_select(%c : !fabric.bits<1>, %a : !fabric.bits<1>, %b : !fabric.bits<1>) {
  fabric.spatial_pe(%pc = %c : !fabric.bits<1>,
                    %pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%cn = %pc : !fabric.bits<1>,
              %x = %pa : !fabric.bits<1>,
              %y = %pb : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@arith.select] (%cn, %x, %y)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
             -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK: func.func private @fu0_subgraph_0
// CHECK: dataflow.subgraph
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i1
// CHECK: dataflow.yield

// No second template emitted (no configurable knobs).
// CHECK-NOT: func.func private @fu0_subgraph_1
