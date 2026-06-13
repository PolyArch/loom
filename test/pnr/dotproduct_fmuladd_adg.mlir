// RUN: loom %s | FileCheck %s

// CHECK-LABEL: fabric.module @dotproduct_fmuladd_adg
fabric.module @dotproduct_fmuladd_adg(%mgr : memref<?x!fabric.bits<32>>,
                                      %i64a : !fabric.bits<64>,
                                      %i64b : !fabric.bits<64>,
                                      %i64c : !fabric.bits<64>,
                                      %init : !fabric.bits<32>,
                                      %ctrl : !fabric.bits<0>) {
  %idx = fabric.pe [spatial] (%pa = %i64a : !fabric.bits<64> to !fabric.bits<32>,
                              %pb = %i64b : !fabric.bits<64> to !fabric.bits<32>,
                              %pc = %i64c : !fabric.bits<64> to !fabric.bits<32>,
                              %lhs = %data0 : !fabric.bits<32>,
                              %rhs = %data1 : !fabric.bits<32>,
                              %acc_init = %init : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>,
              %a = %lhs : !fabric.bits<32>,
              %b = %rhs : !fabric.bits<32>,
              %initial = %acc_init : !fabric.bits<32>)
        -> !fabric.bits<32> {
      // CHECK: fabric.op [@dataflow.stream]
      %stream_idx, %rwc = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
                          {hw_params = [{step_op = ["+="], cont_cond = ["<"]}],
                           sw_configs = {step_op = "+=", cont_cond = "<"}}
                          : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                            -> (!fabric.bits<32>, !fabric.bits<1>)
      // CHECK: fabric.op [@dataflow.carry]
      %carried = fabric.op [@dataflow.carry] (%rwc, %initial, %mac)
                 : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
                   -> !fabric.bits<32>
      // CHECK: fabric.op [@llvm.intr.fmuladd]
      %mac = fabric.op [@llvm.intr.fmuladd] (%a, %b, %carried)
             : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
               -> !fabric.bits<32>
      fabric.yield %stream_idx : !fabric.bits<32>
    }
  }
  // CHECK: fabric.mem [spatial]
  %data0, %done0, %data1, %done1 =
      fabric.mem [spatial] mgr(%mgr) load(%idx, %ctrl, %idx, %ctrl) store()
        [{load_group_size = 2 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>,
           !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>,
            !fabric.bits<32>, !fabric.bits<0>)
  fabric.pe [spatial] (%pa = %done0 : !fabric.bits<0>,
                       %pb = %done1 : !fabric.bits<0>)
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
