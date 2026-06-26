// RUN: loom %s | FileCheck %s

// CHECK: fabric.module @shared_reduction_adg
// CHECK: fabric.op [@arith.extui]
// CHECK: fabric.mem
// CHECK: fabric.switch

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
                    %pd = %stream_sum_lhs : !fabric.bits<32>,
                    %pe = %stream_sum_rhs : !fabric.bits<32>,
                    %pi = %scan_init : !fabric.bits<32>,
                    %pn = %scan_feedback : !fabric.bits<32>,
                    %ps = %scan_scale : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>,
              %sum_lhs = %pd : !fabric.bits<32>,
              %sum_rhs = %pe : !fabric.bits<32>,
              %init = %pi : !fabric.bits<32>,
              %next = %pn : !fabric.bits<32>,
              %scale = %ps : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
      %idx, %rwc = fabric.op [@dataflow.stream] (%fa, %fb, %fc) {hw_params = [{cont_cond = ["<", ">"], step_op = ["+="]}], sw_configs = {cont_cond = "<", step_op = "+="}} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<1>)
      %carried = fabric.op [@dataflow.carry] (%rwc, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %sum = fabric.op [@arith.addi] (%sum_lhs, %sum_rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %stable_scale = fabric.op [@dataflow.invariant] (%rwc, %scale) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %idx : !fabric.bits<32>, %sum : !fabric.bits<32>, %carried : !fabric.bits<32>, %stable_scale : !fabric.bits<32>, %rwc : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %aux_idx, %aux_rwc = fabric.pe [spatial] (%pa = %aux_stream_lb : !fabric.bits<32>,
                    %pb = %aux_stream_ub : !fabric.bits<32>,
                    %pc = %aux_stream_step : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %aux_op_idx, %aux_op_rwc = fabric.op [@dataflow.stream] (%fa, %fb, %fc) {hw_params = [{cont_cond = ["<", ">"], step_op = ["+="]}], sw_configs = {cont_cond = "<", step_op = "+="}} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield %aux_op_idx : !fabric.bits<32>, %aux_op_rwc : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %aux_gate_cond, %aux_active_idx = fabric.pe [spatial] (%pa = %gate_cond : !fabric.bits<32>,
                    %pb = %gate_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %after_cond, %after_value = fabric.op [@dataflow.gate] (%cond, %value) {sw_configs = {value_kind = "data"}} : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield %after_cond : !fabric.bits<1> to !fabric.bits<32>, %after_value : !fabric.bits<32>
    }
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %after_cond, %after_value = fabric.op [@dataflow.gate] (%cond, %value) {sw_configs = {value_kind = "data"}} : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield %after_cond : !fabric.bits<1> to !fabric.bits<32>, %after_value : !fabric.bits<32>
    }
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %after_cond, %after_value = fabric.op [@dataflow.gate] (%cond, %value) {sw_configs = {value_kind = "data"}} : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield %after_cond : !fabric.bits<1> to !fabric.bits<32>, %after_value : !fabric.bits<32>
    }
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %after_cond, %after_value = fabric.op [@dataflow.gate] (%cond, %value) {sw_configs = {value_kind = "data"}} : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield %after_cond : !fabric.bits<1> to !fabric.bits<32>, %after_value : !fabric.bits<32>
    }
  }
  %abs_data = fabric.pe [spatial] (%pa = %data0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %abs = fabric.op [@llvm.intr.abs] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %abs : !fabric.bits<32>
    }
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %abs = fabric.op [@llvm.intr.fabs] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %abs : !fabric.bits<32>
    }
  }
  %squared_data = fabric.pe [spatial] (%pa = %mul_lhs_input : !fabric.bits<32>,
                    %pb = %mul_rhs_input : !fabric.bits<32>) -> !fabric.bits<32> {
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
  %fp_running_aux = fabric.pe [spatial] (%pa = %fp_lhs_aux : !fabric.bits<32>,
                    %pb = %fp_rhs_aux : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  %fp_invariant = fabric.pe [spatial] (%pa = %fp_gate : !fabric.bits<32>,
                    %pb = %fp_invariant_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %bit_invariant = fabric.pe [spatial] (%pa = %fp_gate : !fabric.bits<32>,
                    %pb = %i32d : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %bit_invariant_aux0 = fabric.pe [spatial] (%pa = %fp_gate : !fabric.bits<32>,
                    %pb = %i32c : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %aux_invariant2 = fabric.pe [spatial] (%pa = %aux_invariant_cond : !fabric.bits<32>,
                    %pb = %bit_invariant_aux1_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %bit_invariant_aux1 = fabric.pe [spatial] (%pa = %fp_gate : !fabric.bits<32>,
                    %pb = %bit_invariant_aux1_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %aux_invariant0 = fabric.pe [spatial] (%pa = %aux_invariant_cond : !fabric.bits<32>,
                    %pb = %aux_invariant0_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %aux_invariant1 = fabric.pe [spatial] (%pa = %aux_invariant_cond : !fabric.bits<32>,
                    %pb = %aux_invariant1_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
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
  %fp_diff_aux = fabric.pe [spatial] (%pa = %fp_diff_aux_lhs : !fabric.bits<32>,
                    %pb = %fp_diff_aux_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %diff = fabric.op [@arith.subf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %diff : !fabric.bits<32>
    }
  }
  %scaled_reduction = fabric.pe [spatial] (%pa = %scaled_reduction_lhs : !fabric.bits<32>,
                    %pb = %scaled_reduction_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %product = fabric.op [@arith.mulf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %product : !fabric.bits<32>
    }
  }
  %bit_carry = fabric.pe [spatial] (%pa = %bit_carry_cond : !fabric.bits<32>,
                    %pb = %bit_carry_init : !fabric.bits<32>,
                    %pc = %bit_carry_next : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %init = %pb : !fabric.bits<32>,
              %next = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %carried = fabric.op [@dataflow.carry] (%cond, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %carried : !fabric.bits<32>
    }
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %init = %pb : !fabric.bits<32>,
              %next = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %carried = fabric.op [@dataflow.carry] (%cond, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %carried : !fabric.bits<32>
    }
  }
  %int_sum = fabric.pe [spatial] (%pa = %int_add_lhs : !fabric.bits<32>,
                    %pb = %int_add_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addi, @arith.subi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  %int_product = fabric.pe [spatial] (%pa = %int_mul_lhs : !fabric.bits<32>,
                    %pb = %int_mul_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %product = fabric.op [@arith.muli] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %product : !fabric.bits<32>
    }
  }
  %int_product_aux = fabric.pe [spatial] (%pa = %int_mul_aux_lhs : !fabric.bits<32>,
                    %pb = %int_mul_aux_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %product = fabric.op [@arith.muli] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %product : !fabric.bits<32>
    }
  }
  %int_div0 = fabric.pe [spatial] (%pa = %int_div0_lhs : !fabric.bits<32>,
                    %pb = %int_div0_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %quotient = fabric.op [@arith.divsi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %quotient : !fabric.bits<32>
    }
  }
  %int_div1 = fabric.pe [spatial] (%pa = %int_div1_lhs : !fabric.bits<32>,
                    %pb = %int_div1_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %quotient = fabric.op [@arith.divsi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %quotient : !fabric.bits<32>
    }
  }
  %int_rem = fabric.pe [spatial] (%pa = %int_rem_lhs : !fabric.bits<32>,
                    %pb = %int_rem_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %remainder = fabric.op [@arith.remsi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %remainder : !fabric.bits<32>
    }
  }
  %uint_rem = fabric.pe [spatial] (%pa = %uint_rem_lhs : !fabric.bits<32>,
                    %pb = %uint_rem_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %remainder = fabric.op [@arith.divui, @arith.remui] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %remainder : !fabric.bits<32>
    }
  }
  %fp_div = fabric.pe [spatial] (%pa = %fp_div_lhs : !fabric.bits<32>,
                    %pb = %fp_div_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %quotient = fabric.op [@arith.divf, @arith.remf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %quotient : !fabric.bits<32>
    }
  }
  %addr_shift_const = fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_aux_const = fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_bias_const = fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_extra_const0 = fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_extra_const1 = fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %logic_shifted = fabric.pe [spatial] (%pa = %logic_shift_lhs : !fabric.bits<32>,
                    %pb = %logic_shift_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %shifted = fabric.op [@arith.shrui] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %shifted : !fabric.bits<32>
    }
  }
  %addr_unscaled = fabric.pe [spatial] (%pa = %addr_unscale_lhs : !fabric.bits<32>,
                    %pb = %addr_unscale_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %shifted = fabric.op [@arith.shrui] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %shifted : !fabric.bits<32>
    }
  }
  %addr_shifted = fabric.pe [spatial] (%pa = %addr_shift_lhs : !fabric.bits<32>,
                    %pb = %addr_shift_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %shifted = fabric.op [@arith.shli] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %shifted : !fabric.bits<32>
    }
  }
  %logic_masked = fabric.pe [spatial] (%pa = %logic_mask_lhs : !fabric.bits<32>,
                    %pb = %logic_mask_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %masked = fabric.op [@arith.andi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %masked : !fabric.bits<32>
    }
  }
  %int_or = fabric.pe [spatial] (%pa = %int_or_lhs : !fabric.bits<32>,
                    %pb = %int_or_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %combined = fabric.op [@arith.ori] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %combined : !fabric.bits<32>
    }
  }
  %int_xor = fabric.pe [spatial] (%pa = %int_xor_lhs : !fabric.bits<32>,
                    %pb = %int_xor_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %combined = fabric.op [@arith.xori] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %combined : !fabric.bits<32>
    }
  }
  %packed_sat = fabric.pe [spatial] (%pa = %packed_sat_lhs : !fabric.bits<32>,
                    %pb = %packed_sat_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %packed = fabric.op [@llvm.arm.qadd16, @llvm.arm.qsub16, @llvm.arm.qsub8] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %packed : !fabric.bits<32>
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
  %mac_result1 = fabric.pe [spatial] (%pa = %mac1_lhs : !fabric.bits<32>,
                    %pb = %mac1_rhs : !fabric.bits<32>,
                    %pc = %mac1_acc : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %acc = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %mac = fabric.op [@llvm.intr.fmuladd] (%lhs, %rhs, %acc) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %mac : !fabric.bits<32>
    }
  }
  %unsigned_minmax = fabric.pe [spatial] (%pa = %minmax_lhs : !fabric.bits<32>,
                    %pb = %minmax_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected = fabric.op [@llvm.intr.umax] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected : !fabric.bits<32>
    }
  }
  %rotated = fabric.pe [spatial] (%pa = %rotate_lhs : !fabric.bits<32>,
                    %pb = %rotate_rhs : !fabric.bits<32>,
                    %pc = %rotate_amount : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %amount = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %rotated_value = fabric.op [@llvm.intr.fshl] (%lhs, %rhs, %amount) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %rotated_value : !fabric.bits<32>
    }
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %amount = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %rotated_value = fabric.op [@llvm.intr.fshl] (%lhs, %rhs, %amount) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %rotated_value : !fabric.bits<32>
    }
  }
  %abs = fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %abs = fabric.op [@llvm.intr.abs] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %abs : !fabric.bits<32>
    }
  }
  %swapped = fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %swapped = fabric.op [@llvm.intr.bswap] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %swapped : !fabric.bits<32>
    }
  }
  %cast0_result, %cast1_result, %cast2_result, %cast3_result = fabric.pe [spatial] (%pa = %cast0_input : !fabric.bits<32>,
                    %pb = %cast1_input : !fabric.bits<32>,
                    %pc = %cast2_input : !fabric.bits<32>,
                    %pd = %cast3_input : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%value0 = %pa : !fabric.bits<32>,
              %value1 = %pb : !fabric.bits<32>,
              %value2 = %pc : !fabric.bits<32>,
              %value3 = %pd : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
      %converted0 = fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext] (%value0) : (!fabric.bits<32>) -> !fabric.bits<32>
      %converted1 = fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext] (%value1) : (!fabric.bits<32>) -> !fabric.bits<32>
      %converted2 = fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext] (%value2) : (!fabric.bits<32>) -> !fabric.bits<32>
      %converted3 = fabric.op [@llvm.trunc, @llvm.sext, @llvm.zext] (%value3) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %converted0, %converted1, %converted2, %converted3 : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  %int_extui = fabric.pe [spatial] (%pa = %int_extui_input : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %int_extui = fabric.op [@arith.extui] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %int_extui : !fabric.bits<32>
    }
  }
  %wide_zext0 = fabric.pe [spatial] (%pa = %wide_zext0_input : !fabric.bits<32> to !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%value = %pa : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
      %wide = fabric.op [@llvm.sext, @llvm.zext] (%value) : (!fabric.bits<32>) -> !fabric.bits<64>
      fabric.yield %wide : !fabric.bits<64>
    }
  }
  %wide_zext1 = fabric.pe [spatial] (%pa = %wide_zext1_input : !fabric.bits<32> to !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%value = %pa : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
      %wide = fabric.op [@llvm.sext, @llvm.zext] (%value) : (!fabric.bits<32>) -> !fabric.bits<64>
      fabric.yield %wide : !fabric.bits<64>
    }
  }
  %wide_product = fabric.pe [spatial] (%pa = %wide_mul_lhs : !fabric.bits<64>,
                    %pb = %wide_mul_rhs : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %value = fabric.op [@arith.muli] (%lhs, %rhs) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %value : !fabric.bits<64>
    }
  }
  %wide_remainder = fabric.pe [spatial] (%pa = %wide_rem_lhs : !fabric.bits<64>,
                    %pb = %wide_rem_rhs : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %value = fabric.op [@arith.divui, @arith.remui] (%lhs, %rhs) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %value : !fabric.bits<64>
    }
  }
  %wide_truncated_wide = fabric.pe [spatial] (%pa = %wide_trunc_input : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%value = %pa : !fabric.bits<64>) -> !fabric.bits<64> {
      %narrow = fabric.op [@llvm.trunc] (%value) : (!fabric.bits<64>) -> !fabric.bits<32>
      fabric.yield %narrow : !fabric.bits<32> to !fabric.bits<64>
    }
  }
  %fp = fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %fp = fabric.op [@llvm.uitofp] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %fp : !fabric.bits<32>
    }
  }
  %fp_negated = fabric.pe [spatial] (%pa = %fp_negated_input : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %fp_negated = fabric.op [@llvm.fneg] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %fp_negated : !fabric.bits<32>
    }
  }
  %cmpf_pred = fabric.pe [spatial] (%pa = %cmp_lhs : !fabric.bits<32>,
                    %pb = %cmp_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %pred = fabric.op [@arith.cmpf] (%lhs, %rhs) {hw_params = [{predicate = ["oeq", "ogt", "ugt", "ule", "olt"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %cmpi_pred = fabric.pe [spatial] (%pa = %cmp_lhs : !fabric.bits<32>,
                    %pb = %cmp_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %pred = fabric.op [@arith.cmpi] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %cmpi_pred_aux = fabric.pe [spatial] (%pa = %cmp_lhs : !fabric.bits<32>,
                    %pb = %cmp_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %pred = fabric.op [@arith.cmpi] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %cmpi64_pred = fabric.pe [spatial] (%pa = %cmp64_lhs : !fabric.bits<64>,
                    %pb = %cmp64_rhs : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %pred = fabric.op [@arith.cmpi] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]} : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<64>
    }
  }
  %wide_pred_extui = fabric.pe [spatial] (%pa = %cmpi64_pred : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%value = %pa : !fabric.bits<64> to !fabric.bits<1>) -> !fabric.bits<64> {
      %extended = fabric.op [@arith.extui] (%value) : (!fabric.bits<1>) -> !fabric.bits<64>
      fabric.yield %extended : !fabric.bits<64>
    }
  }
  %selected = fabric.pe [spatial] (%pa = %select_pred : !fabric.bits<32>,
                    %pb = %select_true : !fabric.bits<32>,
                    %pc = %select_false : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %when_true = %pb : !fabric.bits<32>,
              %when_false = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected_value = fabric.op [@arith.select] (%sel, %when_true, %when_false) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected_value : !fabric.bits<32>
    }
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %when_true = %pb : !fabric.bits<32>,
              %when_false = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected_value = fabric.op [@arith.select] (%sel, %when_true, %when_false) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected_value : !fabric.bits<32>
    }
  }
  %control_demux_false, %control_demux_true = fabric.pe [spatial] (%pa = %demux_sel : !fabric.bits<32>,
                    %pb = %demux_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %false_lane, %true_lane = fabric.op [@dataflow.demux] (%sel, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %false_lane, %true_lane : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  %compute_demux_false, %compute_demux_true = fabric.pe [spatial] (%pa = %demux_sel : !fabric.bits<32>,
                    %pb = %demux_then_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %false_lane, %true_lane = fabric.op [@dataflow.demux] (%sel, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %false_lane, %true_lane : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  %control_muxed = fabric.pe [spatial] (%pa = %mux_sel : !fabric.bits<32>,
                    %pb = %mux_false : !fabric.bits<32>,
                    %pc = %mux_true : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %false_lane = %pb : !fabric.bits<32>,
              %true_lane = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected_lane = fabric.op [@dataflow.mux] (%sel, %false_lane, %true_lane) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected_lane : !fabric.bits<32>
    }
  }
  %control_token_demux_false, %control_token_demux_true = fabric.pe [spatial] (%pa = %control_token_demux_sel : !fabric.bits<32>,
                    %pb = %ctrl : !fabric.bits<0> to !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32> to !fabric.bits<0>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %false_lane, %true_lane = fabric.op [@dataflow.demux] (%sel, %value) : (!fabric.bits<1>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %false_lane : !fabric.bits<0> to !fabric.bits<32>, %true_lane : !fabric.bits<0> to !fabric.bits<32>
    }
  }
  %control_token_muxed = fabric.pe [spatial] (%pa = %control_token_mux_sel : !fabric.bits<32>,
                    %pb = %control_token_mux_false : !fabric.bits<0> to !fabric.bits<32>,
                    %pc = %control_token_mux_true : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %false_lane = %pb : !fabric.bits<32> to !fabric.bits<0>,
              %true_lane = %pc : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %selected_lane = fabric.op [@dataflow.mux] (%sel, %false_lane, %true_lane) : (!fabric.bits<1>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
      fabric.yield %selected_lane : !fabric.bits<0> to !fabric.bits<32>
    }
  }
  fabric.pe [spatial] (%pa = %sync_head : !fabric.bits<0>,
                    %pb = %vector_sync_mid : !fabric.bits<0>,
                    %pc = %sync_tail : !fabric.bits<0>,
                    %pd = %sync_extra : !fabric.bits<0>,
                    %pe = %done4 : !fabric.bits<0>,
                    %pf = %sync_lane5 : !fabric.bits<0>,
                    %pg = %store_done0 : !fabric.bits<0>) -> !fabric.bits<0> {
    fabric.fu(%fa = %pa : !fabric.bits<0>,
              %fb = %pb : !fabric.bits<0>,
              %fc = %pc : !fabric.bits<0>,
              %fd = %pd : !fabric.bits<0>,
              %fe = %pe : !fabric.bits<0>,
              %ff = %pf : !fabric.bits<0>,
              %fg = %pg : !fabric.bits<0>) -> !fabric.bits<0> {
      %sync_done0, %sync_done1, %sync_done2, %sync_done3, %sync_done4, %sync_done5, %sync_done6 = fabric.op [@dataflow.sync] (%fa, %fb, %fc, %fd, %fe, %ff, %fg) {sw_configs = {bitmask = "1111111"}} : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
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
  %addr_sum = fabric.pe [spatial] (%pa = %addr_add_lhs : !fabric.bits<32>,
                    %pb = %addr_add_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addi, @arith.subi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  %addr_masked = fabric.pe [spatial] (%pa = %addr_mask_lhs : !fabric.bits<32>,
                    %pb = %addr_mask_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %masked = fabric.op [@arith.andi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %masked : !fabric.bits<32>
    }
  }
  %aux_masked = fabric.pe [spatial] (%pa = %aux_mask_lhs : !fabric.bits<32>,
                    %pb = %aux_mask_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %masked = fabric.op [@arith.andi] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %masked : !fabric.bits<32>
    }
  }
  %aux_xor = fabric.pe [spatial] (%pa = %aux_xor_lhs : !fabric.bits<32>,
                    %pb = %aux_xor_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %xor_value = fabric.op [@arith.xori] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %xor_value : !fabric.bits<32>
    }
  }
  %mac_result2 = fabric.pe [spatial] (%pa = %mac2_lhs : !fabric.bits<32>,
                    %pb = %mac2_rhs : !fabric.bits<32>,
                    %pc = %mac2_acc : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %acc = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %mac = fabric.op [@llvm.intr.fmuladd] (%lhs, %rhs, %acc) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %mac : !fabric.bits<32>
    }
  }
  %mac_result3 = fabric.pe [spatial] (%pa = %mac3_lhs : !fabric.bits<32>,
                    %pb = %mac3_rhs : !fabric.bits<32>,
                    %pc = %mac3_acc : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>,
              %acc = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %mac = fabric.op [@llvm.intr.fmuladd] (%lhs, %rhs, %acc) : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %mac : !fabric.bits<32>
    }
  }
  %state_carry = fabric.pe [spatial] (%pa = %state_carry_cond : !fabric.bits<32>,
                    %pb = %state_carry_init : !fabric.bits<32>,
                    %pc = %state_carry_next : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %init = %pb : !fabric.bits<32>,
              %next = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %carried = fabric.op [@dataflow.carry] (%cond, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %carried : !fabric.bits<32>
    }
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %init = %pb : !fabric.bits<32>,
              %next = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %carried = fabric.op [@dataflow.carry] (%cond, %init, %next) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %carried : !fabric.bits<32>
    }
  }
  %scaled_reduction_aux = fabric.pe [spatial] (%pa = %scaled_reduction_aux_lhs : !fabric.bits<32>,
                    %pb = %scaled_reduction_aux_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %product = fabric.op [@arith.mulf] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %product : !fabric.bits<32>
    }
  }
  %logic_mask_lhs = fabric.switch [spatial] %i32a, %data0, %data1, %bit_carry, %addr_unscaled, %logic_shifted, %int_xor, %aux_xor, %cmpi_pred, %cmpi_pred_aux, %running
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %logic_mask_rhs = fabric.switch [spatial] %i32b, %i32c, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %cmpi_pred, %cmpi_pred_aux
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_mask_lhs = fabric.switch [spatial] %carried_scan, %bit_carry, %state_carry, %selected, %addr_shifted, %running, %idx, %data0, %data1
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_mask_rhs = fabric.switch [spatial] %i32a, %i32b, %i32c, %i32d, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_xor_lhs = fabric.switch [spatial] %selected, %addr_shifted, %addr_unscaled, %logic_shifted, %carried_scan, %bit_carry, %state_carry, %logic_masked, %addr_masked, %int_xor, %aux_masked
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_xor_rhs = fabric.switch [spatial] %carried_scan, %bit_carry, %state_carry, %i32a, %i32b, %data0, %data1, %i32c, %i32d, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1
    [{connectivity_table = ["11111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_add_lhs = fabric.switch [spatial] %i32a, %data1, %data0, %carried_scan, %running, %squared_data, %bit_carry, %reduction_scale, %int_product, %int_product_aux, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %int_extui
    [{connectivity_table = ["1111111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_add_rhs = fabric.switch [spatial] %i32b, %data0, %data1, %fp_invariant, %idx, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %int_rem, %aux_idx, %aux_active_idx, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %int_extui, %int_product, %int_product_aux, %squared_data
    [{connectivity_table = ["11111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_mul_lhs = fabric.switch [spatial] %i32a, %int_xor, %data0, %data1, %int_div0, %int_div1, %aux_idx, %aux_active_idx, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %int_sum, %running
    [{connectivity_table = ["11111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_mul_rhs = fabric.switch [spatial] %i32b, %data0, %data1, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %fp_invariant, %cast0_result, %cast1_result, %cast2_result, %cast3_result
    [{connectivity_table = ["111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_mul_aux_lhs = fabric.switch [spatial] %i32a, %int_xor, %data0, %data1, %int_div0, %int_div1, %aux_idx, %aux_active_idx
    [{connectivity_table = ["11111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_mul_aux_rhs = fabric.switch [spatial] %i32b, %data0, %data1, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %fp_invariant
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_div0_lhs = fabric.switch [spatial] %int_sum, %addr_sum, %aux_idx, %aux_active_idx, %i32b, %i32c
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_div0_rhs = fabric.switch [spatial] %i32c, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_div1_lhs = fabric.switch [spatial] %int_sum, %addr_sum, %aux_idx, %aux_active_idx, %i32b, %i32c
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_div1_rhs = fabric.switch [spatial] %i32c, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_rem_lhs = fabric.switch [spatial] %aux_idx, %aux_active_idx, %i32a, %i32b
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_rem_rhs = fabric.switch [spatial] %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %i32d
    [{connectivity_table = ["11111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %uint_rem_lhs = fabric.switch [spatial] %int_product, %aux_idx, %aux_active_idx, %i32b, %addr_shifted, %running
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %uint_rem_rhs = fabric.switch [spatial] %i32c, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2
    [{connectivity_table = ["11111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_or_lhs = fabric.switch [spatial] %i32a, %logic_masked, %data0, %data1, %addr_shifted
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_or_rhs = fabric.switch [spatial] %i32b, %logic_masked, %data0, %data1
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_xor_lhs = fabric.switch [spatial] %i32a, %rotated, %logic_shifted, %addr_unscaled, %logic_masked, %data0, %packed_sat, %selected, %addr_masked, %aux_masked, %cmpf_pred, %cmpi_pred, %cmpi_pred_aux
    [{connectivity_table = ["1111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_xor_rhs = fabric.switch [spatial] %i32b, %data1, %data0, %logic_masked, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %carried_scan, %bit_carry, %state_carry, %addr_masked, %selected, %aux_masked
    [{connectivity_table = ["111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %packed_sat_lhs = fabric.switch [spatial] %i32a, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %cast0_result, %cast1_result, %cast2_result, %cast3_result
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %packed_sat_rhs = fabric.switch [spatial] %logic_masked, %addr_masked, %data0, %data1, %i32b, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %cast0_result, %cast1_result, %cast2_result, %cast3_result
    [{connectivity_table = ["11111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %minmax_lhs = fabric.switch [spatial] %i32a, %i32b, %data0, %data1, %idx, %running, %int_sum, %addr_sum, %addr_masked, %logic_masked, %carried_scan, %bit_carry, %state_carry, %cast0_result, %cast1_result, %cast2_result, %cast3_result
    [{connectivity_table = ["11111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %minmax_rhs = fabric.switch [spatial] %i32b, %i32c, %data0, %data1, %idx, %running, %int_sum, %addr_sum, %addr_shift_const, %addr_aux_const, %addr_bias_const, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1
    [{connectivity_table = ["1111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %rotate_lhs = fabric.switch [spatial] %i32a, %data1, %data0, %logic_masked, %int_sum, %int_product
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %rotate_rhs = fabric.switch [spatial] %i32b, %data1, %data0, %logic_masked, %int_sum, %int_product
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %rotate_amount = fabric.switch [spatial] %i32c, %data0, %reduction_scale, %addr_shift_const
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %cmp_lhs = fabric.switch [spatial] %i32a, %logic_masked, %data0, %data1, %bit_carry, %running, %addr_sum, %addr_masked, %aux_masked
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %cmp_rhs = fabric.switch [spatial] %i32b, %i32c, %reduction_scale, %data1, %data0, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_masked
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %int_extui_input = fabric.switch [spatial] %i32a, %cmpi_pred, %cmpi_pred_aux, %cmpf_pred, %logic_masked, %addr_masked, %int_xor, %aux_xor
    [{connectivity_table = ["11111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %select_pred = fabric.switch [spatial] %i32a, %logic_masked, %addr_masked, %cmpi_pred, %cmpi_pred_aux, %cmpf_pred, %aux_masked
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %select_true = fabric.switch [spatial] %i32b, %idx, %data1, %rotated, %data0, %int_sum, %addr_sum, %addr_shifted, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %aux_xor, %carried_scan
    [{connectivity_table = ["111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %select_false = fabric.switch [spatial] %i32c, %rotated, %data0, %data1, %carried_scan, %bit_carry, %addr_shift_const, %addr_aux_const, %addr_bias_const, %aux_xor, %running
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %gate_cond = fabric.switch [spatial] %aux_rwc, %logic_masked, %addr_masked, %cmpi_pred, %cmpi_pred_aux, %cmpf_pred, %fp_gate, %i32a
    [{connectivity_table = ["11111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %gate_value = fabric.switch [spatial] %aux_idx, %idx, %running, %addr_sum, %int_sum, %squared_data, %carried_scan, %cast0_result, %cast1_result, %cast2_result, %cast3_result
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %demux_sel = fabric.switch [spatial] %logic_masked, %addr_masked, %cmpi_pred, %cmpi_pred_aux, %cmpf_pred, %fp_gate, %i32a
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %demux_value = fabric.switch [spatial] %carried_scan, %bit_carry, %state_carry, %fp_invariant, %reduction_scale, %running
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %demux_then_value = fabric.switch [spatial] %mac_result, %mac_result1, %mac_result2, %mac_result3, %fp_running, %fp_running_aux, %scaled_reduction, %data0, %data1, %int_sum, %addr_sum, %int_product, %int_product_aux, %selected
    [{connectivity_table = ["11111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mux_sel = fabric.switch [spatial] %logic_masked, %addr_masked, %cmpi_pred, %cmpi_pred_aux, %cmpf_pred, %fp_gate, %i32a
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mux_false = fabric.switch [spatial] %control_demux_false, %carried_scan, %bit_carry, %state_carry, %fp_invariant
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mux_true = fabric.switch [spatial] %compute_demux_true, %mac_result, %mac_result1, %mac_result2, %mac_result3, %fp_running, %fp_running_aux, %scaled_reduction, %data0, %data1
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %control_token_demux_sel = fabric.switch [spatial] %logic_masked, %addr_masked, %cmpi_pred, %cmpi_pred_aux, %cmpf_pred, %fp_gate, %i32a
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %control_token_demux_false_token = fabric.fifo %control_token_demux_false [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %control_token_demux_true_token = fabric.fifo %control_token_demux_true [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %control_token_muxed_token = fabric.fifo %control_token_muxed [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %control_token_mux_sel = fabric.switch [spatial] %logic_masked, %addr_masked, %cmpi_pred, %cmpi_pred_aux, %cmpf_pred, %fp_gate, %i32a
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %control_token_mux_false = fabric.switch [spatial] %control_token_demux_false_token, %store_done0, %ctrl
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %control_token_mux_true = fabric.switch [spatial] %store_done0, %control_token_demux_true_token, %ctrl
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %load1_addr = fabric.switch [spatial] %idx, %i32b, %addr_unscaled, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %running, %addr_sum, %squared_data, %int_sum, %carried_scan, %aux_idx, %aux_active_idx, %selected, %logic_masked, %addr_masked, %int_extui, %addr_shift_const, %addr_aux_const, %addr_bias_const, %addr_extra_const0, %addr_extra_const1
    [{connectivity_table = ["11111111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %cast0_input = fabric.switch [spatial] %i32a, %data0, %data1, %logic_masked, %packed_sat, %idx, %running, %int_sum, %addr_sum, %uint_rem
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %cast1_input = fabric.switch [spatial] %i32a, %data0, %data1, %logic_masked, %packed_sat, %idx, %running, %int_sum, %addr_sum, %uint_rem, %cast0_result
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %cast2_input = fabric.switch [spatial] %i32a, %data0, %data1, %logic_masked, %packed_sat, %idx, %running, %int_sum, %addr_sum, %uint_rem, %cast0_result, %cast1_result
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %cast3_input = fabric.switch [spatial] %i32a, %data0, %data1, %logic_masked, %packed_sat, %idx, %running, %int_sum, %addr_sum, %uint_rem, %cast0_result, %cast1_result, %cast2_result
    [{connectivity_table = ["1111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %wide_zext0_input = fabric.switch [spatial] %data0, %data1, %i32a, %cast0_result, %cast1_result, %unsigned_minmax
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %wide_zext1_input = fabric.switch [spatial] %data1, %data0, %i32b, %cast0_result, %cast1_result
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %wide_mul_lhs = fabric.switch [spatial] %wide_zext1, %wide_zext0, %i64a, %i64b
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %wide_mul_rhs = fabric.switch [spatial] %wide_zext0, %wide_zext1, %i64a, %i64c
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %wide_rem_lhs = fabric.switch [spatial] %wide_product, %wide_zext0, %wide_zext1, %i64a
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %wide_rem_rhs = fabric.switch [spatial] %i64a, %i64b, %i64c, %wide_zext0, %wide_zext1
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %wide_trunc_input = fabric.switch [spatial] %wide_remainder, %wide_product, %wide_zext0, %wide_zext1, %wide_pred_extui
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %cmp64_lhs = fabric.switch [spatial] %i64a, %i64b, %i64c, %wide_zext0, %wide_zext1, %wide_product, %wide_remainder
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %cmp64_rhs = fabric.switch [spatial] %i64a, %i64b, %i64c, %wide_zext0, %wide_zext1, %wide_product, %wide_remainder
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
  %fp_negated_input = fabric.switch [spatial] %data0, %data1, %data2, %data3, %data4, %data5, %fp_running, %fp_running_aux, %fp_diff, %fp_diff_aux, %scaled_reduction
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %load2_addr = fabric.switch [spatial] %i32c, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %idx, %addr_sum, %running, %squared_data, %int_sum, %aux_idx, %aux_active_idx, %data0, %data1, %int_extui, %addr_shift_const, %addr_aux_const, %addr_bias_const, %addr_extra_const0, %addr_extra_const1
    [{connectivity_table = ["11111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %wide_truncated = fabric.fifo %wide_truncated_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %store0_value = fabric.switch [spatial] %scan_store_value, %fp_running, %fp_running_aux, %running, %mac_result, %mac_result1, %mac_result2, %mac_result3, %data0, %data1, %data2, %data3, %data4, %data5, %selected, %rotated, %addr_masked, %logic_masked, %int_xor, %packed_sat, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %abs_data, %scaled_reduction, %scaled_reduction_aux, %int_product, %reduction_scale, %int_sum, %addr_sum, %fp_diff, %fp_diff_aux, %compute_demux_false, %compute_demux_true, %wide_truncated, %fp_negated
    [{connectivity_table = ["1111111111111111111111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %store1_value = fabric.switch [spatial] %i32d, %data0, %data1, %data2, %data3, %data4, %data5, %selected, %scaled_reduction, %scaled_reduction_aux, %mac_result, %mac_result1, %mac_result2, %mac_result3
    [{connectivity_table = ["11111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %vector_sync_mid = fabric.switch [spatial] %done1, %store_done0, %control_token_muxed_token
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %sync_head = fabric.switch [spatial] %done0, %store_done0
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %sync_tail = fabric.switch [spatial] %store_done0, %done2
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %sync_extra = fabric.switch [spatial] %store_done1, %done3, %store_done0
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %sync_lane5 = fabric.switch [spatial] %done5, %store_done0
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %addr_add_lhs = fabric.switch [spatial] %idx, %i32a, %i32b, %i32c, %squared_data, %int_product, %running, %reduction_scale, %int_product_aux, %data0, %data1
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %addr_add_rhs = fabric.switch [spatial] %fp_invariant, %reduction_scale, %i32a, %i32b, %idx, %int_rem, %aux_idx, %aux_active_idx, %carried_scan, %int_product, %int_product_aux, %squared_data
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> !fabric.bits<32>
  %addr_mask_lhs = fabric.switch [spatial] %addr_sum, %idx, %data0, %data1, %logic_masked, %carried_scan, %bit_carry, %state_carry, %selected, %aux_masked, %aux_xor
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %addr_mask_rhs = fabric.switch [spatial] %reduction_scale, %fp_invariant, %i32b, %i32c, %int_xor, %packed_sat, %logic_masked, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_masked, %aux_xor
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %addr_unscale_lhs = fabric.switch [spatial] %i32a, %addr_shifted, %bit_carry, %data0, %squared_data, %int_product, %int_product_aux
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %addr_unscale_rhs = fabric.switch [spatial] %i32b, %addr_shift_const, %reduction_scale, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %logic_shift_lhs = fabric.switch [spatial] %i32a, %data0, %data1, %carried_scan, %bit_carry, %state_carry, %running, %addr_unscaled, %addr_shifted, %logic_masked, %addr_masked, %int_xor, %aux_xor
    [{connectivity_table = ["1111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %logic_shift_rhs = fabric.switch [spatial] %i32b, %addr_shifted, %reduction_scale, %addr_shift_const, %addr_aux_const, %addr_bias_const, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %addr_shift_lhs = fabric.switch [spatial] %i32a, %carried_scan, %idx, %bit_carry, %state_carry, %selected, %aux_masked, %aux_xor
    [{connectivity_table = ["11111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %addr_shift_rhs = fabric.switch [spatial] %i32b, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %addr_shift_const, %addr_aux_const, %addr_bias_const, %aux_masked, %aux_xor, %aux_invariant0, %aux_invariant1, %aux_invariant2
    [{connectivity_table = ["1111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %load0_addr = fabric.switch [spatial] %idx, %addr_masked, %addr_shifted, %addr_unscaled, %carried_scan, %bit_carry, %state_carry, %squared_data, %running, %addr_sum, %int_product, %int_sum, %aux_idx, %aux_active_idx, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %selected, %addr_shift_const, %addr_aux_const, %addr_bias_const, %int_extui
    [{connectivity_table = ["11111111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %load3_addr = fabric.switch [spatial] %i32d, %carried_scan, %idx, %squared_data, %running, %addr_sum, %int_sum, %aux_idx, %aux_active_idx, %int_extui
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %load4_addr = fabric.switch [spatial] %idx, %squared_data, %running, %addr_sum, %int_product, %int_sum, %addr_unscaled, %addr_shifted, %aux_idx, %aux_active_idx, %int_extui
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %load5_addr = fabric.switch [spatial] %idx, %squared_data, %running, %addr_sum, %int_product, %int_sum, %addr_unscaled, %addr_shifted, %aux_idx, %aux_active_idx, %int_extui
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %store0_addr = fabric.switch [spatial] %idx, %addr_unscaled, %carried_scan, %addr_shift_const, %state_carry, %addr_aux_const, %addr_bias_const, %addr_extra_const0, %addr_extra_const1, %int_sum, %addr_sum, %aux_idx, %running, %aux_active_idx, %control_demux_false, %control_demux_true, %int_extui
    [{connectivity_table = ["11111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %store1_addr = fabric.switch [spatial] %i32c, %idx, %addr_unscaled, %carried_scan, %addr_shift_const, %addr_aux_const, %addr_bias_const, %addr_extra_const0, %addr_extra_const1, %int_sum, %addr_sum, %aux_idx, %running, %aux_active_idx, %int_extui
    [{connectivity_table = ["111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_stream_lb = fabric.switch [spatial] %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_stream_ub = fabric.switch [spatial] %int_product, %int_product_aux, %squared_data
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_stream_step = fabric.switch [spatial] %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_invariant_cond = fabric.switch [spatial] %aux_rwc, %fp_gate
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_invariant0_value = fabric.switch [spatial] %i32a, %i32b, %i32c, %i32d, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %fp_invariant, %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %aux_invariant1_value = fabric.switch [spatial] %i32a, %i32b, %i32c, %i32d, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %fp_invariant, %aux_invariant0, %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["1111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %store0_ctrl = fabric.switch [spatial] %ctrl, %done0, %done1, %done2, %done3, %done4, %done5, %control_token_demux_false_token, %control_token_demux_true_token
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %store1_ctrl = fabric.switch [spatial] %ctrl, %done0, %done1, %done2, %done3, %done4, %done5, %control_token_demux_false_token, %control_token_demux_true_token
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  %data0, %done0, %data1, %done1, %data2, %done2, %data3, %done3, %data4, %done4, %data5, %done5, %store_done0, %store_done1 =
      fabric.mem [spatial] mgr(%mgr) load(%load0_addr, %ctrl, %load1_addr, %ctrl, %load2_addr, %ctrl, %load3_addr, %ctrl, %load4_addr, %ctrl, %load5_addr, %ctrl)
                                store(%store0_addr, %store0_value, %store0_ctrl, %store1_addr, %store1_value, %store1_ctrl)
        [{load_group_size = 6 : i32, store_group_size = 2 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %mul_lhs_input = fabric.switch [spatial] %data0, %data1, %data2, %idx, %data4, %int_div0, %int_div1, %aux_idx, %aux_active_idx, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %aux_invariant0, %aux_invariant1, %aux_invariant2, %int_sum, %running
    [{connectivity_table = ["111111111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mul_rhs_input = fabric.switch [spatial] %data0, %data1, %data2, %data4, %reduction_scale, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %fp_invariant, %cast0_result, %cast1_result, %cast2_result, %cast3_result
    [{connectivity_table = ["1111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %reduction_input = fabric.switch [spatial] %data0, %abs_data, %squared_data
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> !fabric.bits<32>
  %stream_sum_lhs = fabric.switch [spatial] %reduction_input, %carried_scan, %bit_carry, %state_carry, %int_product, %int_product_aux, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %int_extui
    [{connectivity_table = ["11111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %stream_sum_rhs = fabric.switch [spatial] %carried_scan, %fp_invariant, %reduction_scale, %bit_invariant_aux1, %int_rem, %aux_idx, %aux_invariant0, %aux_invariant1, %aux_invariant2, %bit_invariant, %bit_invariant_aux0, %cast0_result, %cast1_result, %cast2_result, %cast3_result, %addr_shifted, %int_extui
    [{connectivity_table = ["11111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scan_init = fabric.switch [spatial] %i32a, %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scan_scale = fabric.switch [spatial] %i32b, %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_lhs = fabric.switch [spatial] %carried_scan, %data0, %data2, %data4, %reduction_scale, %mac_result1
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_rhs = fabric.switch [spatial] %data0, %data1, %data3, %data5, %reduction_scale
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_lhs_aux = fabric.switch [spatial] %bit_carry, %state_carry, %carried_scan, %data0, %data1, %data2, %data4, %reduction_scale, %mac_result1
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_rhs_aux = fabric.switch [spatial] %data1, %data0, %data3, %data5, %reduction_scale, %fp_invariant, %bit_invariant
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_diff_lhs = fabric.switch [spatial] %i32a, %data0
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_diff_rhs = fabric.switch [spatial] %i32b, %fp_invariant, %data1, %fp_div
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_diff_aux_lhs = fabric.switch [spatial] %data1, %data0, %i32a
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_diff_aux_rhs = fabric.switch [spatial] %bit_invariant, %fp_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %i32b, %data1, %fp_div
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_div_lhs = fabric.switch [spatial] %data1, %data0
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_div_rhs = fabric.switch [spatial] %data2, %fp_invariant, %reduction_scale
    [{connectivity_table = ["111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %fp_invariant_value = fabric.switch [spatial] %i32b, %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %bit_invariant_aux1_value = fabric.switch [spatial] %i32b, %reduction_scale, %addr_shift_const, %addr_aux_const, %addr_bias_const
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scaled_reduction_lhs = fabric.switch [spatial] %carried_scan, %fp_running, %fp_running_aux, %data1, %data3, %data5, %data0, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %fp_negated, %reduction_scale
    [{connectivity_table = ["111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scaled_reduction_rhs = fabric.switch [spatial] %reduction_scale, %data4, %data5, %data1, %data3, %state_carry, %bit_carry, %aux_invariant0, %aux_invariant1, %aux_invariant2, %fp_negated, %data0
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scaled_reduction_aux_lhs = fabric.switch [spatial] %carried_scan, %fp_running, %fp_running_aux, %data1, %data3, %data5, %data0, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %aux_invariant0, %aux_invariant1, %aux_invariant2, %fp_negated, %reduction_scale
    [{connectivity_table = ["111111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scaled_reduction_aux_rhs = fabric.switch [spatial] %reduction_scale, %data4, %data5, %data1, %data3, %state_carry, %bit_carry, %aux_invariant0, %aux_invariant1, %aux_invariant2, %fp_negated, %data0
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac_lhs = fabric.switch [spatial] %i32a, %data0, %data2, %data4, %fp_diff, %fp_diff_aux, %scaled_reduction, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %data1, %bit_invariant_aux1, %reduction_scale
    [{connectivity_table = ["1111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac_rhs = fabric.switch [spatial] %i32b, %data1, %data2, %data3, %data5, %fp_diff, %fp_diff_aux, %data0, %bit_carry, %state_carry
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac_acc = fabric.switch [spatial] %i32c, %carried_scan, %bit_carry, %scaled_reduction, %state_carry, %data0
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac1_lhs = fabric.switch [spatial] %i32a, %data2, %data4, %data0, %fp_diff, %fp_diff_aux, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %reduction_scale
    [{connectivity_table = ["11111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac1_rhs = fabric.switch [spatial] %i32b, %data3, %data5, %data1, %fp_diff, %fp_diff_aux, %bit_carry, %state_carry, %carried_scan
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac1_acc = fabric.switch [spatial] %i32c, %mac_result, %scaled_reduction, %carried_scan, %bit_carry, %state_carry
    [{connectivity_table = ["111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac2_lhs = fabric.switch [spatial] %i32a, %data0, %data2, %data4, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %reduction_scale
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac2_rhs = fabric.switch [spatial] %i32b, %data0, %data1, %data3, %data5, %bit_carry, %state_carry, %carried_scan
    [{connectivity_table = ["11111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac2_acc = fabric.switch [spatial] %mac_result1, %mac_result, %scaled_reduction, %bit_carry, %state_carry
    [{connectivity_table = ["11111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac3_lhs = fabric.switch [spatial] %i32a, %data0, %data2, %data4, %fp_invariant, %bit_invariant, %bit_invariant_aux0, %bit_invariant_aux1, %reduction_scale
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac3_rhs = fabric.switch [spatial] %i32b, %data0, %data1, %data3, %data5, %bit_carry, %state_carry, %carried_scan, %fp_running, %fp_running_aux
    [{connectivity_table = ["1111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %mac3_acc = fabric.switch [spatial] %mac_result2, %mac_result1, %mac_result, %scaled_reduction, %bit_carry, %state_carry, %data4
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %bit_carry_cond = fabric.switch [spatial] %i32a, %fp_gate
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %bit_carry_init = fabric.switch [spatial] %i32b, %i32c, %addr_shift_const, %addr_bias_const
    [{connectivity_table = ["1111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %bit_carry_next = fabric.switch [spatial] %i32c, %addr_unscaled, %mac_result, %mac_result1, %int_sum, %selected, %running, %mac_result2, %mac_result3, %data0, %state_carry, %aux_masked, %aux_xor, %fp_running_aux
    [{connectivity_table = ["11111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %state_carry_cond = fabric.switch [spatial] %fp_gate, %i32a
    [{connectivity_table = ["11"]}]
    : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %state_carry_init = fabric.switch [spatial] %i32a, %i32b, %i32c, %i32d, %addr_shift_const, %addr_aux_const, %addr_bias_const, %data0, %data1
    [{connectivity_table = ["111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %state_carry_next = fabric.switch [spatial] %mac_result, %mac_result1, %mac_result2, %mac_result3, %bit_carry, %carried_scan, %int_sum, %data0, %running, %aux_masked, %aux_xor, %fp_running_aux
    [{connectivity_table = ["111111111111"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
  %scan_feedback, %scan_store_value = fabric.switch [spatial] %running, %fp_running, %mac_result, %mac_result1, %mac_result2, %mac_result3, %bit_carry, %state_carry, %int_or, %selected, %int_sum, %addr_sum, %int_product, %int_product_aux, %control_muxed, %int_xor, %aux_masked, %aux_xor, %fp_running_aux, %uint_rem
    [{connectivity_table = ["11111111111111111111", "00111100000000000000"]}]
    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>)
  %sync_aux_done = fabric.switch [spatial] %store_done0, %done1, %done2, %done3, %done4, %done5, %control_token_muxed_token
    [{connectivity_table = ["1111111"]}]
    : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
  fabric.yield
}
