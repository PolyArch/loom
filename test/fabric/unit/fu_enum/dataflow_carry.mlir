// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU containing dataflow.carry. The op has a fixed shape (i1 cond + 2x T)
// so the only knobs come from outer mux/demux.

// CHECK-LABEL: @fu_carry
func.func @fu_carry(%cond: !fabric.bits<1>,
                    %init: !fabric.bits<32>,
                    %carry: !fabric.bits<32>) {
  %r = fabric.fu(%c = %cond : !fabric.bits<1>,
                 %i = %init : !fabric.bits<32>,
                 %k = %carry : !fabric.bits<32>) -> !fabric.bits<32> {
    %o = fabric.op [@dataflow.carry] (%c, %i, %k)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
           -> !fabric.bits<32>
    fabric.yield %o : !fabric.bits<32>
  }

  // CHECK: dataflow.carry %{{.*}}, %{{.*}}, %{{.*}} : i32

  return
}
