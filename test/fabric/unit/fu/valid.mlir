// RUN: loom %s | loom | FileCheck %s

// Minimal FU: one fabric.op, FU has no outputs. The enclosing PE still
// needs L >= 1 so we declare a single bits<W> output that the FU does
// not drive (PE outputs are wired to FU outputs at config time).
// CHECK-LABEL: fabric.module @fu_min
fabric.module @fu_min(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // CHECK: fabric.pe
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    // CHECK: fabric.fu(%{{.*}} = %{{.*}} : !fabric.bits<32>, %{{.*}} = %{{.*}} : !fabric.bits<32>) -> ()
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> () {
      // CHECK: fabric.op
      %0 = fabric.op [@arith.muli] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// FU yielding one value, with mux feeding the op.
// CHECK-LABEL: fabric.module @fu_mux_op_yield
fabric.module @fu_mux_op_yield(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    // CHECK: %{{.*}} = fabric.fu
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>,
                   %z = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      // CHECK: fabric.mux
      %m = fabric.mux %x, %y, %z {sel = 1 : i32, discard = false, disconnect = false}
           : !fabric.bits<32>
      // CHECK: fabric.op
      %k = fabric.op [@arith.addi] (%m, %z)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// FU with op then demux fanning out two values.
// CHECK-LABEL: fabric.module @fu_op_demux
fabric.module @fu_op_demux(%a : !fabric.bits<16>, %b : !fabric.bits<16>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<16>,
                    %pb = %b : !fabric.bits<16>)
                   -> (!fabric.bits<16>, !fabric.bits<16>) {
    // CHECK: %{{.*}}:2 = fabric.fu
    %r:2 = fabric.fu(%x = %pa : !fabric.bits<16>, %y = %pb : !fabric.bits<16>)
                    -> (!fabric.bits<16>, !fabric.bits<16>) {
      %k = fabric.op [@arith.muli] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      // CHECK: fabric.demux
      %d0, %d1 = fabric.demux %k {sel = 0 : i32, discard = false, disconnect = false}
                 : !fabric.bits<16> -> 2
      fabric.yield %d0, %d1 : !fabric.bits<16>, !fabric.bits<16>
    }
  }
  fabric.yield
}

// FU output boundary widening: inner yield value bits<1> with outer
// FU result bits<32>. Hardware zero-fills the high 31 bits at the FU
// boundary so the value reaching the PE port is bits<32>, satisfying
// the PE-uniform-width invariant.
// CHECK-LABEL: fabric.module @fu_yield_widen
fabric.module @fu_yield_widen(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    // CHECK: fabric.fu(%{{.*}} = %{{.*}} : !fabric.bits<32>, %{{.*}} = %{{.*}} : !fabric.bits<32>) -> !fabric.bits<32>
    %r = fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %p = fabric.op [@arith.cmpi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerCompareMinMax>, hw_params = {integer_widths = [32 : i32], predicates = ["eq", "ne"]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      // CHECK: fabric.yield %{{.*}} : !fabric.bits<1> to !fabric.bits<32>
      fabric.yield %p : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  fabric.yield
}

// FU with multiple fabric.op nodes whose connectivity could be reconfigured by
// inner mux/demux selectors.
// CHECK-LABEL: fabric.module @fu_multi_op
fabric.module @fu_multi_op(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    // CHECK: %{{.*}} = fabric.fu
    %r = fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      // CHECK: fabric.op [@arith.addi, @arith.subi]
      %s = fabric.op [@arith.addi, @arith.subi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %t = fabric.op [@arith.muli] (%s, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %t : !fabric.bits<32>
    }
  }
  fabric.yield
}
