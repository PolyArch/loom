// RUN: loom %s | FileCheck %s

// CHECK-LABEL: fabric.module @byte_swap_store_adg
fabric.module @byte_swap_store_adg(%mgr : memref<?x!fabric.bits<32>>,
                                   %idx : !fabric.bits<32>,
                                   %ctrl : !fabric.bits<0>) {
  // CHECK: fabric.mem [spatial]
  %sub, %data, %load_done =
      fabric.mem [spatial] mgr(%mgr) load(%idx, %ctrl) store()
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
        -> (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
  %swapped = fabric.pe [spatial] (%input = %data : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%value = %input : !fabric.bits<32>) -> !fabric.bits<32> {
      // CHECK: fabric.op [@llvm.intr.bswap]
      %result = fabric.op [@llvm.intr.bswap] (%value)
                : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %result : !fabric.bits<32>
    }
  }
  // CHECK: fabric.mem [spatial]
  %store_done =
      fabric.mem [spatial] mgr(%sub) load() store(%idx, %swapped, %ctrl)
        [{load_group_size = 0 : i32, store_group_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<0>) -> !fabric.bits<0>
  fabric.pe [spatial] (%pa = %load_done : !fabric.bits<0>,
                       %pb = %store_done : !fabric.bits<0>)
      -> !fabric.bits<0> {
    fabric.fu(%fa = %pa : !fabric.bits<0>,
              %fb = %pb : !fabric.bits<0>)
        -> !fabric.bits<0> {
      // CHECK: fabric.op [@dataflow.sync]
      %sa, %sb = fabric.op [@dataflow.sync] (%fa, %fb)
                {sw_configs = {bitmask = "11"}}
                : (!fabric.bits<0>, !fabric.bits<0>)
                  -> (!fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %sa : !fabric.bits<0>
    }
  }
  fabric.yield
}
