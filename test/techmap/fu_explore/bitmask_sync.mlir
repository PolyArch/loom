// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// Pins: variadic dataflow.sync with M=3 hardware ports plus a
// hw_params allowed-set restriction of three bitmasks. Each bitmask
// materializes one dataflow.sync template whose port count equals the
// bitmask popcount N.

// CHECK-LABEL: @fu_sync3
fabric.module @fu_sync3(%cast0_fu_sync3 : !fabric.bits<32>, %cast1_fu_sync3 : !fabric.bits<32>, %cast2_fu_sync3 : !fabric.bits<32>) {
  fabric.pe [spatial] (%a = %cast0_fu_sync3 : !fabric.bits<32>, %b = %cast1_fu_sync3 : !fabric.bits<32>, %c = %cast2_fu_sync3 : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
  %x, %y, %z = fabric.fu(%aa = %a : !fabric.bits<32>,
                         %bb = %b : !fabric.bits<32>,
                         %cc = %c : !fabric.bits<32>)
                        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    %i, %j, %k = fabric.op [@dataflow.sync] (%aa, %bb, %cc)
                 {hw_params = [{bitmask = ["110", "011", "111"]}]}
                 : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %i, %j, %k : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
  }
  // CHECK-DAG: dataflow.sync
  }
  fabric.yield
}

