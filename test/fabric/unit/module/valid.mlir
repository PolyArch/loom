// RUN: loom %s | loom | FileCheck %s

// Empty module: parser inserts an implicit fabric.yield terminator and the
// printer omits the implicit terminator on the round-trip.
// CHECK-LABEL: fabric.module @m_empty
// CHECK-NEXT: }
fabric.module @m_empty {
}

// Module with an explicit fabric.yield terminator. The implicit-terminator
// printer still elides the yield, so the body round-trips as empty.
// CHECK-LABEL: fabric.module @m_explicit_yield
// CHECK-NEXT: }
fabric.module @m_explicit_yield {
  fabric.yield
}

// Module body holding the canonical fabric containers (spatial_pe, fifo).
// CHECK-LABEL: fabric.module @m_with_inner_ops
// CHECK: fabric.spatial_pe
// CHECK: fabric.fu
// CHECK: fabric.op
// CHECK: fabric.fifo
fabric.module @m_with_inner_ops {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %r = fabric.spatial_pe(%pa = %a : !fabric.bits<32>,
                         %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  %f = fabric.fifo %a [max_depth = 4, bypassable = false] : !fabric.bits<32>
  fabric.yield
}

// Two distinct modules in one input file: each carries its own sym_name and
// each round-trips independently.
// CHECK-LABEL: fabric.module @m_first
// CHECK-LABEL: fabric.module @m_second
fabric.module @m_first {
}
fabric.module @m_second {
}
