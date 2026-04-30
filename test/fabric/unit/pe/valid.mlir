// RUN: loom %s | loom | FileCheck %s

// Minimal pe: K=1, L=1, single inner FU.
// CHECK-LABEL: fabric.module @pe_min
fabric.module @pe_min(%a : !fabric.bits<32>) {
  // CHECK: fabric.pe [spatial] (%{{.*}} = %{{.*}} : !fabric.bits<32>) -> !fabric.bits<32>
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    // CHECK: fabric.fu
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// Two-port pe with a single FU consuming both inputs and producing two
// PE-level results (FU outputs are not SSA-wired to PE results).
// CHECK-LABEL: fabric.module @pe_2x2
fabric.module @pe_2x2(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // CHECK: %{{.*}}:2 = fabric.pe
  %r:2 = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                           %pb = %b : !fabric.bits<32>)
                          -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

// Heterogeneous PE with two inner FUs of different shapes (K=2, L=1).
// CHECK-LABEL: fabric.module @pe_heterogeneous
fabric.module @pe_heterogeneous(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // CHECK: fabric.pe
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                         %pb = %b : !fabric.bits<32>)
                        -> !fabric.bits<32> {
    // CHECK: fabric.fu
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %v = fabric.op [@arith.addi] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
    // CHECK: fabric.fu
    fabric.fu(%ga = %pa : !fabric.bits<32>) -> (!fabric.bits<32>) {
      %w = fabric.op [@math.absi] (%ga)
           : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %w : !fabric.bits<32>
    }
  }
  fabric.yield
}

// Boundary case: max_fu_inputs == K and max_fu_outputs == L.
// CHECK-LABEL: fabric.module @pe_boundary
fabric.module @pe_boundary(%a : !fabric.bits<16>, %b : !fabric.bits<16>) {
  // CHECK: fabric.pe
  %r:2 = fabric.pe [spatial] (%pa = %a : !fabric.bits<16>,
                           %pb = %b : !fabric.bits<16>)
                          -> (!fabric.bits<16>, !fabric.bits<16>) {
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>)
              -> (!fabric.bits<16>, !fabric.bits<16>) {
      %v = fabric.op [@arith.muli] (%fa, %fb)
           : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      %d:2 = fabric.demux %v {sel = 0 : i32, discard = false, disconnect = false}
             : !fabric.bits<16> -> 2
      fabric.yield %d#0, %d#1 : !fabric.bits<16>, !fabric.bits<16>
    }
  }
  fabric.yield
}

// Module containing two distinct pe ops; each round-trips.
// CHECK-LABEL: fabric.module @pe_two_pes
// CHECK: fabric.pe
// CHECK: fabric.pe
fabric.module @pe_two_pes(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  %r0 = fabric.pe [spatial] (%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %pa : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
  }
  %r1 = fabric.pe [spatial] (%qa = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fa = %qa : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %w = fabric.op [@arith.muli] (%fa, %fa)
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %w : !fabric.bits<8>
    }
  }
  fabric.yield
}
