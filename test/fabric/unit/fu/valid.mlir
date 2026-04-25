// RUN: loom %s | loom | FileCheck %s

// Minimal FU: one fabric.op, FU has no outputs.
// CHECK-LABEL: @fu_min
func.func @fu_min(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  // CHECK: fabric.fu(%{{.*}} = %{{.*}} : !fabric.bits<32>, %{{.*}} = %{{.*}} : !fabric.bits<32>) -> ()
  fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>) -> () {
    // CHECK: fabric.op
    %0 = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield
  }
  return
}

// FU yielding one value, with mux feeding the op.
// CHECK-LABEL: @fu_mux_op_yield
func.func @fu_mux_op_yield(%a: !fabric.bits<32>, %b: !fabric.bits<32>, %c: !fabric.bits<32>)
    -> !fabric.bits<32> {
  // CHECK: %{{.*}} = fabric.fu
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    // CHECK: fabric.mux
    %m = fabric.mux %x, %y, %z {sel = 1 : i32, discard = false, disconnect = false}
         : !fabric.bits<32>
    // CHECK: fabric.op
    %k = fabric.op [@arith.addi] (%m, %z)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return %r : !fabric.bits<32>
}

// FU with op then demux fanning out two values.
// CHECK-LABEL: @fu_op_demux
func.func @fu_op_demux(%a: !fabric.bits<16>, %b: !fabric.bits<16>)
    -> (!fabric.bits<16>, !fabric.bits<16>) {
  // CHECK: %{{.*}}:2 = fabric.fu
  %r:2 = fabric.fu(%x = %a : !fabric.bits<16>, %y = %b : !fabric.bits<16>)
                  -> (!fabric.bits<16>, !fabric.bits<16>) {
    %k = fabric.op [@arith.muli] (%x, %y)
         : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
    // CHECK: fabric.demux
    %d0, %d1 = fabric.demux %k {sel = 0 : i32, discard = false, disconnect = false}
               : !fabric.bits<16> -> 2
    fabric.yield %d0, %d1 : !fabric.bits<16>, !fabric.bits<16>
  }
  return %r#0, %r#1 : !fabric.bits<16>, !fabric.bits<16>
}

// FU with multiple fabric.op nodes whose connectivity could be reconfigured by
// inner mux/demux selectors.
// CHECK-LABEL: @fu_multi_op
func.func @fu_multi_op(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  // CHECK: %{{.*}} = fabric.fu
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    // CHECK: fabric.op [@arith.addi, @arith.subi]
    %s = fabric.op [@arith.addi, @arith.subi] (%x, %y)
         {sw_configs = {op_sel = "arith.subi"}}
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %t = fabric.op [@arith.muli] (%s, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %t : !fabric.bits<32>
  }
  return %r : !fabric.bits<32>
}
