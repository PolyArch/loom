// RUN: loom-cost-test %s | FileCheck %s

// Each FU pins one implementation-family representative at width 32; the printed
// score must equal `baseUnit * (bw/32)` exactly.
//   addi    -> baseUnit 1.0  -> 1.0
//   andi    -> baseUnit 0.5  -> 0.5
//   addf    -> baseUnit 4.0  -> 4.0
//   muli    -> baseUnit 3.0  -> 3.0

fabric.module @bu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @bu_andi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.andi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerLogic>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @bu_addf(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.addf] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarFloatAddSub>, hw_params = {float_formats = ["f32"], behavior = {rounding_modes = ["to_nearest_even"], nan_behaviors = ["ieee"], subnormal_behaviors = ["preserve"], signed_zero_behaviors = ["preserve"], fastmath = "none"}}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @bu_muli_singleton(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.muli] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK: cost bu_addi=1.000000e+00
// CHECK-NEXT: cost bu_andi=5.000000e-01
// CHECK-NEXT: cost bu_addf=4.000000e+00
// CHECK-NEXT: cost bu_muli_singleton=3.000000e+00
