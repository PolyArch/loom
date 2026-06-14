// RUN: loom %s | FileCheck %s

// Checked-in projection of the ADG Builder shared-reduction recipe. Keep this
// fixture synchronized with `loom-adg-builder-test --shared-reduction`; mapping
// tests use it as the reusable shared Fabric ADG, not as a per-workload ADG.
// CHECK-LABEL: fabric.module @shared_reduction_adg
fabric.module @shared_reduction_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %i64a : !fabric.bits<64>,
                                    %i64b : !fabric.bits<64>,
                                    %i64c : !fabric.bits<64>,
                                    %i32a : !fabric.bits<32>,
                                    %i32b : !fabric.bits<32>,
                                    %i32c : !fabric.bits<32>,
                                    %i32d : !fabric.bits<32>,
                                    %ctrl : !fabric.bits<0>) {
  %idx, %running, %carried_scan, %reduction_scale, %fp_gate = fabric.pe [spatial] (%pa = %i64a : !fabric.bits<64> to !fabric.bits<32>,
                    %pb = %i64b : !fabric.bits<64> to !fabric.bits<32>,
                    %pc = %i64c : !fabric.bits<64> to !fabric.bits<32>,
                    %pd = %reduction_input : !fabric.bits<32>,
                    %pi = %i32a : !fabric.bits<32>,
                    %pn = %scan_feedback : !fabric.bits<32>,
                    %ps = %i32b : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>,
              %data = %pd : !fabric.bits<32>,
              %init = %pi : !fabric.bits<32>,
              %next = %pn : !fabric.bits<32>,
              %scale = %ps : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
      %idx, %rwc = fabric.op [@dataflow.stream] (%fa, %fb, %fc) {hw_params = [{cont_cond = ["<"], step_op = ["+="]}], sw_configs = {cont_cond = "<", step_op = "+="}} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<1>)
      %carried = fabric.op [@dataflow.carry] (%rwc, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %sum = fabric.op [@arith.addi] (%data, %carried) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %stable_scale = fabric.op [@dataflow.invariant] (%rwc, %scale) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %idx : !fabric.bits<32>, %sum : !fabric.bits<32>, %carried : !fabric.bits<32>, %stable_scale : !fabric.bits<32>, %rwc : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %abs_data = fabric.pe [spatial] (%pa = %data0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %abs = fabric.op [@llvm.intr.abs] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %abs : !fabric.bits<32>
    }
  }
  %squared_data = fabric.pe [spatial] (%pa = %mul_lhs_input : !fabric.bits<32>,
                    %pb = %data0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %product = fabric.op [@arith.muli] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %product : !fabric.bits<32>
    }
  }
  %fp_running = fabric.pe [spatial] (%pa = %fp_lhs : !fabric.bits<32>,
                    %pb = %fp_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  %fp_invariant = fabric.pe [spatial] (%pa = %fp_gate : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> () {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  %fp_diff = fabric.pe [spatial] (%pa = %fp_diff_lhs : !fabric.bits<32>,
                    %pb = %fp_diff_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %diff = fabric.op [@arith.subf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %diff : !fabric.bits<32>
    }
  }
  %scaled_reduction = fabric.pe [spatial] (%pa = %carried_scan : !fabric.bits<32>,
                    %pb = %reduction_scale : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %product = fabric.op [@arith.mulf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %product : !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>,
                    %pc = %i32c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %init = %pb : !fabric.bits<32>,
              %next = %pc : !fabric.bits<32>) -> () {
      %carried = fabric.op [@dataflow.carry] (%cond, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %init = %pb : !fabric.bits<32>,
              %next = %pc : !fabric.bits<32>) -> () {
      %carried = fabric.op [@dataflow.carry] (%cond, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %sum = fabric.op [@arith.addi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %product = fabric.op [@arith.muli] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %shifted = fabric.op [@arith.shrui] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %shifted = fabric.op [@arith.shli] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %masked = fabric.op [@arith.andi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %combined = fabric.op [@arith.ori] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %combined = fabric.op [@arith.xori] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  %mac_result = fabric.pe [spatial] (%pa = %mac_lhs : !fabric.bits<32>,
                    %pb = %mac_rhs : !fabric.bits<32>,
                    %pc = %mac_acc : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %acc = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %mac = fabric.op [@llvm.intr.fmuladd] (%lhs, %rhs, %acc) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %mac : !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>,
                    %pc = %i32c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %amount = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %rotated = fabric.op [@llvm.intr.fshl] (%lhs, %rhs, %amount) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %rotated : !fabric.bits<32>
    }
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %amount = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %rotated = fabric.op [@llvm.intr.fshl] (%lhs, %rhs, %amount) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %rotated : !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %abs = fabric.op [@llvm.intr.abs] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %abs : !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %swapped = fabric.op [@llvm.intr.bswap] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %swapped : !fabric.bits<32>
    }
  }
  %zext_index = fabric.pe [spatial] (%pa = %zext_input : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %wide = fabric.op [@llvm.zext] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %wide : !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %fp = fabric.op [@llvm.uitofp] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %fp : !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %pred = fabric.op [@arith.cmpf] (%lhs, %rhs) {hw_params = [{predicate = ["oeq", "ogt", "ugt", "ule"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> () {
      %pred = fabric.op [@arith.cmpi] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sgt", "ult", "ule"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield
    }
  }
  fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                    %pb = %i32b : !fabric.bits<32>,
                    %pc = %i32c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %when_true = %pb : !fabric.bits<32>,
              %when_false = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected = fabric.op [@arith.select] (%sel, %when_true, %when_false) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected : !fabric.bits<32>
    }
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %when_true = %pb : !fabric.bits<32>,
              %when_false = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected = fabric.op [@arith.select] (%sel, %when_true, %when_false) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected : !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %done0 : !fabric.bits<0>,
                    %pb = %vector_sync_mid : !fabric.bits<0>,
                    %pc = %sync_tail : !fabric.bits<0>) -> !fabric.bits<0> {
    fabric.fu(%fa = %pa : !fabric.bits<0>,
              %fb = %pb : !fabric.bits<0>,
              %fc = %pc : !fabric.bits<0>) -> !fabric.bits<0> {
      %sync_done0, %sync_done1, %sync_done2 = fabric.op [@dataflow.sync] (%fa, %fb, %fc) {sw_configs = {bitmask = "111"}} : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %sync_done0 : !fabric.bits<0>
    }
  }
  fabric.pe [spatial] (%pc = %done0 : !fabric.bits<0>,
                    %pd = %sync_aux_done : !fabric.bits<0>) -> !fabric.bits<0> {
    fabric.fu(%fc = %pc : !fabric.bits<0>,
              %fd = %pd : !fabric.bits<0>) -> !fabric.bits<0> {
      %sync_done0, %sync_done1 = fabric.op [@dataflow.sync] (%fc, %fd) {sw_configs = {bitmask = "11"}} : (!fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %sync_done0 : !fabric.bits<0>
    }
  }
  %load1_addr = fabric.switch [spatial] %idx, %i32b
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %zext_input = fabric.switch [spatial] %i32a, %data1
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %load2_addr = fabric.switch [spatial] %i32c, %zext_index
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %store0_value = fabric.switch [spatial] %scan_store_value, %fp_running, %running, %mac_result
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> !fabric.bits<32>
  %vector_sync_mid = fabric.switch [spatial] %done1, %store_done0
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %sync_tail = fabric.switch [spatial] %store_done0, %done2
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %data0, %done0, %data1, %done1, %data2, %done2, %data3, %done3, %store_done0, %store_done1 =
      fabric.mem [spatial] mgr(%mgr) load(%idx, %ctrl, %load1_addr, %ctrl, %load2_addr, %ctrl, %i32d, %ctrl)
                                store(%idx, %store0_value, %ctrl, %i32c, %i32d, %ctrl)
        [{load_group_size = 4 : i32, store_group_size = 2 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %mul_lhs_input = fabric.switch [spatial] %data0, %data1, %data2
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> !fabric.bits<32>
  %reduction_input = fabric.switch [spatial] %data0, %abs_data, %squared_data
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> !fabric.bits<32>
  %fp_lhs = fabric.switch [spatial] %carried_scan, %data0
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_rhs = fabric.switch [spatial] %data0, %data1
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_diff_lhs = fabric.switch [spatial] %i32a, %data0
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_diff_rhs = fabric.switch [spatial] %i32b, %fp_invariant
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac_lhs = fabric.switch [spatial] %i32a, %data0, %fp_diff
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> !fabric.bits<32>
  %mac_rhs = fabric.switch [spatial] %i32b, %data1, %fp_diff
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> !fabric.bits<32>
  %mac_acc = fabric.switch [spatial] %i32c, %carried_scan
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scan_feedback, %scan_store_value = fabric.switch [spatial] %running, %fp_running, %mac_result
    [{connectivity_table = ["111", "110"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>)
  %sync_aux_done = fabric.switch [spatial] %store_done0, %done1, %done2, %done3
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
    -> !fabric.bits<0>
  fabric.yield
}
