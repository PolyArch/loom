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

func.func @cost_2port(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                      %c: !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.mux %x, %y : !fabric.bits<32>
    %k = fabric.op [@arith.addi] (%m, %z)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return %r : !fabric.bits<32>
}

func.func @cost_4port(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                      %c: !fabric.bits<32>, %d: !fabric.bits<32>,
                      %e: !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>,
                 %z = %c : !fabric.bits<32>,
                 %w = %d : !fabric.bits<32>,
                 %v = %e : !fabric.bits<32>) -> !fabric.bits<32> {
    %m = fabric.mux %x, %y, %z, %w : !fabric.bits<32>
    %k = fabric.op [@arith.addi] (%m, %v)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return %r : !fabric.bits<32>
}

// CHECK: cost cost_2port=9.700000e+01
// CHECK-NEXT: cost cost_4port=1.930000e+02
