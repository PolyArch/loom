// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with arith.cmpf and a hardware-supported predicate set. Note: f32
// inputs are encoded as fabric.bits<32>; the enumerator lifts to f32 since
// arith.cmpf's flavor is FloatCmp.

// CHECK-LABEL: @fu_cmpf
func.func @fu_cmpf(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<1> {
    %k = fabric.op [@arith.cmpf] (%x, %y)
         {hw_params = [{predicate = ["oeq", "olt", "ogt"]}]}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
    fabric.yield %k : !fabric.bits<1>
  }

  // CHECK-DAG: arith.cmpf oeq, %{{.*}}, %{{.*}} : f32
  // CHECK-DAG: arith.cmpf olt, %{{.*}}, %{{.*}} : f32
  // CHECK-DAG: arith.cmpf ogt, %{{.*}}, %{{.*}} : f32

  return
}
