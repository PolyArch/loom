// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with arith.cmpf and a hardware-supported predicate set. The FU's
// external boundary uses bits<32> -> bits<1> mapping internally, but to
// satisfy the pe uniform-W rule we expose the cmpf at bits<1>
// throughout (the TypeParam(0) data ports accept any width). The
// enumerator lifts the f-typed cmp via the FloatCmp flavor.

// CHECK-LABEL: fabric.module @fu_cmpf
fabric.module @fu_cmpf(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.cmpf] (%x, %y)
           {hw_params = [{predicate = ["oeq", "olt", "ogt"]}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      %ext = fabric.op [@arith.uitofp] (%k)
             : (!fabric.bits<1>) -> !fabric.bits<32>
      fabric.yield %ext : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-DAG: arith.cmpf oeq, %{{.*}}, %{{.*}} : f32
// CHECK-DAG: arith.cmpf olt, %{{.*}}, %{{.*}} : f32
// CHECK-DAG: arith.cmpf ogt, %{{.*}}, %{{.*}} : f32
