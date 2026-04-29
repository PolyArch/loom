// RUN: loom-template-dump %s | FileCheck %s

// Two FUs in the same module. TemplateLibrary collects both and assigns a
// monotonically increasing global id; ordering across FUs must be stable.

fabric.module @fu_a(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @fu_b(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.muli] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK: tpl#0 root=arith.addi size=1 cfg=
// CHECK-NEXT: tpl#1 root=arith.muli size=1 cfg=
