// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with multi-member float arith group {addf, subf}: lifts fabric.bits<32>
// to f32 and emits arith.addf / arith.subf as the two enumerated subgraphs.

// CHECK-LABEL: fabric.module @fu_addf_or_subf
fabric.module @fu_addf_or_subf(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addf, @arith.subf] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-DAG: arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-DAG: arith.subf %{{.*}}, %{{.*}} : f32
