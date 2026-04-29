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

// Module body holding ops permitted in the broader fabric dialect today.
// (The strict body whitelist is enforced by a later task; for now the body
//  is permissive so unrelated ops parse cleanly.)
// CHECK-LABEL: fabric.module @m_with_inner_ops
// CHECK: fabric.fu
// CHECK: fabric.op
fabric.module @m_with_inner_ops {
  %a = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %b = builtin.unrealized_conversion_cast to !fabric.bits<32>
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
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
