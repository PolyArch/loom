// RUN: loom %s | loom | FileCheck %s

// All test programs share one builtin top-level module so cross-references
// between fabric.module symbols are visible.

// Top-level fabric.module callable from any other site.
// CHECK-LABEL: fabric.module @callee_top
fabric.module @callee_top(%a : !fabric.bits<32>) -> (!fabric.bits<32>) {
  fabric.yield %a : !fabric.bits<32>
}

// Top-level instantiation of a top-level fabric.module symbol. The
// fabric.instantiate appears directly in the builtin top-level module.
// CHECK: fabric.instantiate @callee_top
%t = builtin.unrealized_conversion_cast to !fabric.bits<32>
%u = fabric.instantiate @callee_top(%t : !fabric.bits<32>) -> (!fabric.bits<32>)

// Sibling fabric.module that another module's body will instantiate.
fabric.module @leaf(%x : !fabric.bits<32>) -> (!fabric.bits<32>) {
  fabric.yield %x : !fabric.bits<32>
}

// fabric.module body instantiates a sibling top-level fabric.module.
// CHECK-LABEL: fabric.module @host_calls_leaf
// CHECK: fabric.instantiate @leaf
fabric.module @host_calls_leaf(%a : !fabric.bits<32>) {
  %r = fabric.instantiate @leaf(%a : !fabric.bits<32>) -> (!fabric.bits<32>)
  fabric.yield
}

// Named fabric.pe defined inside a fabric.module body, then instantiated
// later in the same body.
// CHECK-LABEL: fabric.module @named_pe_host
// CHECK: fabric.pe @ALU [spatial]
// CHECK: fabric.instantiate @ALU
fabric.module @named_pe_host(%a : !fabric.bits<32>) {
  %r = fabric.pe @ALU [spatial] (%pa = %a : !fabric.bits<32>)
                              -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  %s = fabric.instantiate @ALU(%a : !fabric.bits<32>) -> (!fabric.bits<32>)
  fabric.yield
}

// Named fabric.fu defined inside a fabric.pe body, then instantiated later
// in the same pe body.
// CHECK-LABEL: fabric.module @named_fu_host
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu @F
// CHECK: fabric.instantiate @F
fabric.module @named_fu_host(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu @F(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    %g = fabric.instantiate @F(%pa : !fabric.bits<32>) -> (!fabric.bits<32>)
  }
  fabric.yield
}

// Width-relaxation on the input direction: SSA operand is bits<32> while
// the callee declares its input as bits<16>. Round-trip preserves the
// `to <inner-type>` clause.
fabric.module @leaf_narrow(%x : !fabric.bits<16>) -> (!fabric.bits<16>) {
  fabric.yield %x : !fabric.bits<16>
}
// CHECK-LABEL: fabric.module @host_relax
// CHECK: fabric.instantiate @leaf_narrow(%{{.*}} : !fabric.bits<32> to !fabric.bits<16>) -> !fabric.bits<16>
fabric.module @host_relax(%a : !fabric.bits<32>) {
  %r = fabric.instantiate @leaf_narrow(%a : !fabric.bits<32>
                                          to !fabric.bits<16>)
       -> (!fabric.bits<16>)
  fabric.yield
}
