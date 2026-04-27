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

// CHECK-LABEL: @fu_select
func.func @fu_select(%c: !fabric.bits<1>,
                     %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%cn = %c : !fabric.bits<1>,
                 %x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %o = fabric.op [@arith.select] (%cn, %x, %y)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %o : !fabric.bits<32>
  }
  return
}

// CHECK: func.func private @fu0_subgraph_0
// CHECK: dataflow.subgraph
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK: dataflow.yield

// No second template emitted (no configurable knobs).
// CHECK-NOT: func.func private @fu0_subgraph_1
