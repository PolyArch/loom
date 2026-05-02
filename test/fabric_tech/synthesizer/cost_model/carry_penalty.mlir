// RUN: loom-cost-test %s | FileCheck %s

// Adding a fabric.op[@dataflow.carry] to an FU strictly increases its
// cost (under positive cost.carry_penalty).
//
// Default carry_penalty = 2.0.
// dataflow.carry schema: (i1 cond, T init, T carry) -> T.
//
// cost_no_carry  = arith.addi i32 baseUnit 1.0 = 1.0
// cost_with_carry = 1.0 (addi) + 2.0 * 32 (carry penalty * bw) = 65.0

func.func @cost_no_carry(%a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%x = %a : !fabric.bits<32>, %y = %b : !fabric.bits<32>)
                -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %k : !fabric.bits<32>
  }
  return %r : !fabric.bits<32>
}

func.func @cost_with_carry(%cond: !fabric.bits<1>, %a: !fabric.bits<32>, %b: !fabric.bits<32>) -> !fabric.bits<32> {
  %r = fabric.fu(%cc = %cond : !fabric.bits<1>,
                 %x = %a : !fabric.bits<32>,
                 %y = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %k = fabric.op [@arith.addi] (%x, %y)
         : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %c = fabric.op [@dataflow.carry] (%cc, %k, %k)
         : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    fabric.yield %c : !fabric.bits<32>
  }
  return %r : !fabric.bits<32>
}

// CHECK: cost cost_no_carry=1.000000e+00
// CHECK-NEXT: cost cost_with_carry=6.500000e+01
