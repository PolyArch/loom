// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU containing a single arith.cmpi fabric.op whose hardware supports three
// predicate values. The enumerator should produce three dataflow.subgraphs,
// one per predicate, each with a properly typed CmpIPredicateAttr.
//
// To satisfy the pe uniform-W rule, the FU exposes a bits<1>
// boundary that mirrors the cmpi result width. The FU's input ports are
// also bits<1>; an internal dataflow.constant materializes a bits<32>
// pattern that feeds the cmpi inputs. The cmpi result is yielded as the
// FU's bits<1> output.

// CHECK-LABEL: fabric.module @fu_cmpi
fabric.module @fu_cmpi(%a : !fabric.bits<1>, %b : !fabric.bits<1>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<1>,
                    %pb = %b : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%x = %pa : !fabric.bits<1>, %y = %pb : !fabric.bits<1>) -> !fabric.bits<1> {
      %k = fabric.op [@arith.cmpi] (%x, %y)
           {hw_params = [{predicate = ["eq", "slt", "sgt"]}]}
           : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
      fabric.yield %k : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK-DAG: "op#0{predicate=eq}"
// CHECK-DAG: "op#0{predicate=slt}"
// CHECK-DAG: "op#0{predicate=sgt}"

// CHECK-DAG: arith.cmpi eq, %{{.*}}, %{{.*}} : i1
// CHECK-DAG: arith.cmpi slt, %{{.*}}, %{{.*}} : i1
// CHECK-DAG: arith.cmpi sgt, %{{.*}}, %{{.*}} : i1
