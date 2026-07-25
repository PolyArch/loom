// RUN: loom-cost-test %s | FileCheck %s

// A 2-port fabric.mux must cost strictly less than a 4-port fabric.mux
// of the same width (under positive cost.mux_penalty).
//
// Default mux_penalty = 1.5; bw = 32.
//   mux contribution = 1.5 * portCount * bw
//   2-port mux: 1.5 *  2 * 32 =  96.0
//   4-port mux: 1.5 *  4 * 32 = 192.0
// Each FU also has one arith.addi i32 (baseUnit 1.0): +1.0.
//   cost_2port = 96 + 1 =  97.0
//   cost_4port = 192 + 1 = 193.0

fabric.module @cost_2port(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                          %c: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>,
                       %pc = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %m = fabric.mux %x, %y : !fabric.bits<32>
      %k = fabric.op [@arith.addi] (%m, %z)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @cost_4port(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                          %c: !fabric.bits<32>, %d: !fabric.bits<32>,
                          %e: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>,
                       %pc = %c : !fabric.bits<32>,
                       %pd = %d : !fabric.bits<32>,
                       %pe = %e : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>,
              %w = %pd : !fabric.bits<32>,
              %v = %pe : !fabric.bits<32>) -> !fabric.bits<32> {
      %m = fabric.mux %x, %y, %z, %w : !fabric.bits<32>
      %k = fabric.op [@arith.addi] (%m, %v)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK: cost cost_2port=9.700000e+01
// CHECK-NEXT: cost cost_4port=1.930000e+02
