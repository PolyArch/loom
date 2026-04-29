// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU containing dataflow.carry. The op has a fixed shape (i1 cond + 2x T)
// so the only knobs come from outer mux/demux. To satisfy the spatial_pe
// uniform-W rule we expose the FU at bits<1> throughout.

// CHECK-LABEL: fabric.module @fu_carry
fabric.module @fu_carry(%cond : !fabric.bits<1>, %init : !fabric.bits<1>, %carry : !fabric.bits<1>) {
  fabric.spatial_pe(%pcond = %cond : !fabric.bits<1>,
                    %pinit = %init : !fabric.bits<1>,
                    %pcarry = %carry : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%c = %pcond : !fabric.bits<1>,
              %i = %pinit : !fabric.bits<1>,
              %k = %pcarry : !fabric.bits<1>) -> !fabric.bits<1> {
      %o = fabric.op [@dataflow.carry] (%c, %i, %k)
           : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
             -> !fabric.bits<1>
      fabric.yield %o : !fabric.bits<1>
    }
  }
  fabric.yield
}

// CHECK: dataflow.carry %{{.*}}, %{{.*}}, %{{.*}} : i1
