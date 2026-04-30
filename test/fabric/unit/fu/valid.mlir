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
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
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
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
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
           : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      // CHECK: fabric.demux
      %d0, %d1 = fabric.demux %k {sel = 0 : i32, discard = false, disconnect = false}
                 : !fabric.bits<16> -> 2
      fabric.yield %d0, %d1 : !fabric.bits<16>, !fabric.bits<16>
    }
  }
  fabric.yield
}

// FU boundary truncation: outer operand bits<32> with inner block-arg
// bits<0>. Hardware drops the high 32 bits at the FU boundary. The outer
// type matches the enclosing PE's uniform W=32, while the inner body
// op (dataflow.constant) consumes the bits<0> none-token.
// CHECK-LABEL: fabric.module @fu_boundary_trunc
fabric.module @fu_boundary_trunc(%ctrl : !fabric.bits<32>) {
  fabric.pe [spatial] (%pctrl = %ctrl : !fabric.bits<32>) -> !fabric.bits<32> {
    // CHECK: fabric.fu(%{{.*}} = %{{.*}} : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32>
    %r = fabric.fu(%c = %pctrl : !fabric.bits<32> to !fabric.bits<0>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@dataflow.constant] (%c)
           {sw_configs = {const_hex_value = "0xdeadbeef"}}
           : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
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
           {sw_configs = {op_sel = "arith.subi"}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %t = fabric.op [@arith.muli] (%s, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %t : !fabric.bits<32>
    }
  }
  fabric.yield
}
