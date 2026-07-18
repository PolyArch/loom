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

// Named fabric.pe defined inside a fabric.module body as a TEMPLATE
// (no SSA results in the host scope), then instantiated later in the
// same body.
// CHECK-LABEL: fabric.module @named_pe_host
// CHECK: fabric.pe @ALU [spatial] (!fabric.bits<32>) -> !fabric.bits<32>
// CHECK: fabric.instantiate @ALU
fabric.module @named_pe_host(%a : !fabric.bits<32>) {
  fabric.pe @ALU [spatial] (!fabric.bits<32>) -> (!fabric.bits<32>) {
  ^bb0(%pa: !fabric.bits<32>):
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    fabric.yield %pa : !fabric.bits<32>
  }
  %s = fabric.instantiate @ALU(%a : !fabric.bits<32>) -> (!fabric.bits<32>)
  fabric.yield
}

// Named fabric.fu defined inside a fabric.pe body as a TEMPLATE, then
// instantiated later in the same pe body.
// CHECK-LABEL: fabric.module @named_fu_host
// CHECK: fabric.pe [spatial]
// CHECK: fabric.fu @F (!fabric.bits<32>) -> !fabric.bits<32>
// CHECK: fabric.instantiate @F
fabric.module @named_fu_host(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu @F (!fabric.bits<32>) -> (!fabric.bits<32>) {
    ^bb0(%fa: !fabric.bits<32>):
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    %g = fabric.instantiate @F(%pa : !fabric.bits<32>) -> (!fabric.bits<32>)
  }
  fabric.yield
}

// Named switch and memory templates are module-level physical resources and
// can be instantiated like named PE templates.
// CHECK-LABEL: fabric.module @named_switch_host
// CHECK: fabric.switch @SW [spatial]
// CHECK: fabric.instantiate @SW
fabric.module @named_switch_host(%a : !fabric.bits<32>,
                                 %b : !fabric.bits<32>) {
  fabric.switch @SW [spatial]
       (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
       [{connectivity_table = ["11", "11"]}]
  %out:2 = fabric.instantiate @SW(%a : !fabric.bits<32>,
                                  %b : !fabric.bits<32>)
           -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// CHECK-LABEL: fabric.module @named_mem_host
// CHECK: fabric.mem @MEM [spatial]
// CHECK: fabric.instantiate @MEM
fabric.module @named_mem_host(%mgr : memref<?x!fabric.bits<32>>,
                              %addr : !fabric.bits<32>,
                              %ctrl : !fabric.bits<0>) {
  fabric.mem @MEM [spatial]
       (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
       [{load_group_size = 1 : i32, store_group_size = 0 : i32,
         data_width = 32 : i32}]
  %data, %done = fabric.instantiate @MEM(
       %mgr : memref<?x!fabric.bits<32>>,
       %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>)
       -> (!fabric.bits<32>, !fabric.bits<0>)
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
