// RUN: loom-cost-test %s | FileCheck %s

// CostModel must scale per-op base cost linearly with bitwidth: an i64
// arith.addi has exactly 2x the cost of an i32 arith.addi.
//
// arith.addi share-group baseUnit = 1.0; baseArea = baseUnit * (bw/32).
//   cost_32 = 1.0 * (32/32) = 1.0
//   cost_64 = 1.0 * (64/32) = 2.0

fabric.module @cost_32(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @cost_64(%a: !fabric.bits<64>, %b: !fabric.bits<64>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<64>,
                       %pb = %b : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%x = %pa : !fabric.bits<64>, %y = %pb : !fabric.bits<64>)
              -> !fabric.bits<64> {
      %k = fabric.op [@arith.addi] (%x, %y)
           : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %k : !fabric.bits<64>
    }
  }
  fabric.yield
}

// CHECK: cost cost_32=1.000000e+00
// CHECK-NEXT: cost cost_64=2.000000e+00
