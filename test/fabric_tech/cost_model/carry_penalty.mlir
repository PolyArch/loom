// RUN: loom-cost-test %s | FileCheck %s

// Adding a fabric.op[@dataflow.carry] to an FU strictly increases its
// cost (under positive cost.carry_penalty).
//
// Default carry_penalty = 2.0.
// dataflow.carry schema: (i1 cond, T init, T carry) -> T.
//
// cost_no_carry  = arith.addi i32 baseUnit 1.0 = 1.0
// cost_with_carry = 1.0 (addi) + 2.0 * 32 (carry penalty * bw) = 65.0

fabric.module @cost_no_carry(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @cost_with_carry(%cond: !fabric.bits<32>, %a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pcond = %cond : !fabric.bits<32>,
                       %pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cc = %pcond : !fabric.bits<32> to !fabric.bits<1>,
              %x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %c = fabric.op [@dataflow.carry] (%cc, %k, %k)
           {implementation_family = #fabric.implementation_family<LoopCarry>, hw_params = {}}
           : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %c : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK: cost cost_no_carry=1.000000e+00
// CHECK-NEXT: cost cost_with_carry=6.500000e+01
