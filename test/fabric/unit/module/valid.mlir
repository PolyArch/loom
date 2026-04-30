// RUN: loom %s | loom | FileCheck %s

// Empty module (no inputs, no outputs): parser inserts an implicit
// fabric.yield terminator and the printer omits the implicit terminator
// on the round-trip.
// CHECK-LABEL: fabric.module @m_empty
// CHECK-SAME: ()
// CHECK-NEXT: }
fabric.module @m_empty() {
}

// Module with an explicit fabric.yield terminator. The implicit-terminator
// printer still elides the yield, so the body round-trips as empty.
// CHECK-LABEL: fabric.module @m_explicit_yield
// CHECK-SAME: ()
// CHECK-NEXT: }
fabric.module @m_explicit_yield() {
  fabric.yield
}

// Module body holding the canonical fabric containers (pe, fifo).
// CHECK-LABEL: fabric.module @m_with_inner_ops
// CHECK-SAME: (%{{.*}}: !fabric.bits<32>, %{{.*}}: !fabric.bits<32>)
// CHECK: fabric.pe
// CHECK: fabric.fu
// CHECK: fabric.op
// CHECK: fabric.fifo
fabric.module @m_with_inner_ops(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
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
fabric.module @m_first() {
}
fabric.module @m_second() {
}

// Module with declared output types. The yield value types match the
// declared output types exactly (no width relaxation needed).
// CHECK-LABEL: fabric.module @m_with_outputs
// CHECK-SAME: -> (!fabric.bits<32>, !fabric.bits<32>)
fabric.module @m_with_outputs(%a : !fabric.bits<32>, %b : !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                         %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield %r, %a : !fabric.bits<32>, !fabric.bits<32>
}

// memref input/output: must round-trip exactly (no width relaxation).
// CHECK-LABEL: fabric.module @m_memref_passthrough
// CHECK-SAME: (%{{.*}}: memref<8xi32>) -> memref<8xi32>
fabric.module @m_memref_passthrough(%mem : memref<8xi32>) -> (memref<8xi32>) {
  fabric.yield %mem : memref<8xi32>
}

// Width relaxation at module-input -> pe operand: the source is
// !fabric.bits<32> and the PE block-arg / inner type is !fabric.bits<16>.
// CHECK-LABEL: fabric.module @m_pe_input_width_relax
// CHECK: fabric.pe [spatial] (%{{.*}} = %{{.*}} : !fabric.bits<32> to !fabric.bits<16>) -> !fabric.bits<16>
fabric.module @m_pe_input_width_relax(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32> to !fabric.bits<16>)
                        -> !fabric.bits<16> {
    fabric.fu(%fa = %pa : !fabric.bits<16>) -> !fabric.bits<16> {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield %v : !fabric.bits<16>
    }
  }
  fabric.yield
}

// Width relaxation at fifo operand: input is !fabric.bits<32>, FIFO inner
// width is !fabric.bits<16>. Round-trip preserves the `to` clause.
// CHECK-LABEL: fabric.module @m_fifo_input_width_relax
// CHECK: fabric.fifo %{{.*}} [max_depth = 4, bypassable = false] : !fabric.bits<32> to !fabric.bits<16>
fabric.module @m_fifo_input_width_relax(%a : !fabric.bits<32>) {
  %0 = fabric.fifo %a [max_depth = 4, bypassable = false]
       : !fabric.bits<32> to !fabric.bits<16>
  fabric.yield
}

// Width relaxation at module yield: source !fabric.bits<32> is yielded for a
// declared !fabric.bits<16> module result, low-bit alignment.
// CHECK-LABEL: fabric.module @m_yield_width_relax
// CHECK-SAME: -> !fabric.bits<16>
// CHECK: fabric.yield %{{.*}} : !fabric.bits<32> to !fabric.bits<16>
fabric.module @m_yield_width_relax(%a : !fabric.bits<32>)
    -> (!fabric.bits<16>) {
  fabric.yield %a : !fabric.bits<32> to !fabric.bits<16>
}
