// RUN: loom-cost-test %s | FileCheck %s

// Each FU pins one share-group representative at width 32; the printed
// score must equal `baseUnit * (bw/32)` exactly.
//   addi    -> baseUnit 1.0  -> 1.0
//   andi    -> baseUnit 0.5  -> 0.5
//   divsi   -> baseUnit 8.0  -> 8.0
//   addf    -> baseUnit 4.0  -> 4.0
//   exp     -> baseUnit 12.0 -> 12.0
//   sqrt    -> baseUnit 8.0  -> 8.0
//   muli    -> singleton fallback baseUnit 1.0 -> 1.0

fabric.module @bu_addi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
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
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @bu_divsi(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@arith.divsi] (%x, %y)
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
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @bu_exp(%a: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@math.exp] (%x) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @bu_sqrt(%a: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %k = fabric.op [@math.sqrt] (%x) : (!fabric.bits<32>) -> !fabric.bits<32>
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
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK: cost bu_addi=1.000000e+00
// CHECK-NEXT: cost bu_andi=5.000000e-01
// CHECK-NEXT: cost bu_divsi=8.000000e+00
// CHECK-NEXT: cost bu_addf=4.000000e+00
// CHECK-NEXT: cost bu_exp=1.200000e+01
// CHECK-NEXT: cost bu_sqrt=8.000000e+00
// CHECK-NEXT: cost bu_muli_singleton=1.000000e+00
