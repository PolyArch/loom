// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with multi-member math group {sin, cos}.

// CHECK-LABEL: fabric.module @fu_sin_or_cos
fabric.module @fu_sin_or_cos(%a : !fabric.bits<32>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@math.sin, @math.cos] (%x)
           : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-DAG: math.sin %{{.*}} : f32
// CHECK-DAG: math.cos %{{.*}} : f32
