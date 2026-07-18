// RUN: loom %s | FileCheck %s

// CHECK: fabric.module @shared_reduction_adg
// CHECK-DAG: fabric.op [@llvm.intr.ctlz]
// CHECK-DAG: fabric.op [@arith.extui]
// CHECK-DAG: fabric.op [@arith.index_cast]
// CHECK-DAG: fabric.op [@arith.cmpi, @llvm.icmp]
// CHECK-DAG: fabric.fifo
// CHECK-DAG: fabric.op [@llvm.select]
// CHECK-DAG: fabric.mem
// CHECK-DAG: fabric.switch

fabric.module @shared_reduction_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %i64a : !fabric.bits<64>,
                                    %i64b : !fabric.bits<64>,
                                    %i64c : !fabric.bits<64>,
                                    %i32a : !fabric.bits<32>,
                                    %i32b : !fabric.bits<32>,
                                    %i32c : !fabric.bits<32>,
                                    %i32d : !fabric.bits<32>,
                                    %ctrl : !fabric.bits<0>) {
  %idx, %running, %carried_scan, %reduction_scale, %fp_gate = fabric.pe [spatial] (%pa = %transport_fanout0_out0 : !fabric.bits<64> to !fabric.bits<32>,
                    %pb = %transport_fanout1_out0 : !fabric.bits<64> to !fabric.bits<32>,
                    %pc = %transport_fanout2_out0 : !fabric.bits<64> to !fabric.bits<32>,
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
      %idx, %rwc = fabric.op [@dataflow.stream] (%fa, %fb, %fc) {hw_params = [{step_kind = 0 : i32, predicate = [2 : i64, 4 : i64]}]} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<1>)
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
      %aux_op_idx, %aux_op_rwc = fabric.op [@dataflow.stream] (%fa, %fb, %fc) {hw_params = [{step_kind = 0 : i32, predicate = [2 : i64, 4 : i64]}]} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield %aux_op_idx : !fabric.bits<32>, %aux_op_rwc : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %aux_gate_cond, %aux_active_idx = fabric.pe [spatial] (%pa = %transport_fanout3_out0 : !fabric.bits<32>,
                    %pb = %gate_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %after_cond, %after_value = fabric.op [@dataflow.gate] (%cond, %value) {sw_configs = {value_kind = "data"}} : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield %after_cond : !fabric.bits<1> to !fabric.bits<32>, %after_value : !fabric.bits<32>
    }
  }
  %aux_gate_cond1, %aux_active_idx1 = fabric.pe [spatial] (%pa = %transport_fanout3_out1 : !fabric.bits<32>,
                    %pb = %gate_value1 : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
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
  %abs_data = fabric.pe [spatial] (%pa = %transport_fanout4_out0 : !fabric.bits<32>) -> !fabric.bits<32> {
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
  %fp_invariant = fabric.pe [spatial] (%pa = %transport_fanout5_out0 : !fabric.bits<32>,
                    %pb = %fp_invariant_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %bit_invariant = fabric.pe [spatial] (%pa = %transport_fanout6_out0 : !fabric.bits<32>,
                    %pb = %bit_invariant_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %bit_invariant_aux0 = fabric.pe [spatial] (%pa = %transport_fanout6_out1 : !fabric.bits<32>,
                    %pb = %bit_invariant_aux0_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %aux_invariant2 = fabric.pe [spatial] (%pa = %transport_fanout6_out2 : !fabric.bits<32>,
                    %pb = %transport_fanout7_out0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %bit_invariant_aux1 = fabric.pe [spatial] (%pa = %transport_fanout5_out1 : !fabric.bits<32>,
                    %pb = %transport_fanout7_out1 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %aux_invariant0 = fabric.pe [spatial] (%pa = %transport_fanout6_out3 : !fabric.bits<32>,
                    %pb = %aux_invariant0_value : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %stable = fabric.op [@dataflow.invariant] (%cond, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %stable : !fabric.bits<32>
    }
  }
  %aux_invariant1 = fabric.pe [spatial] (%pa = %transport_fanout6_out4 : !fabric.bits<32>,
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
  %addr_shift_const = fabric.pe [spatial] (%pa = %transport_fanout8_out0 : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_aux_const = fabric.pe [spatial] (%pa = %transport_fanout8_out1 : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_bias_const = fabric.pe [spatial] (%pa = %transport_fanout8_out2 : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_extra_const0 = fabric.pe [spatial] (%pa = %transport_fanout8_out3 : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %addr_extra_const1 = fabric.pe [spatial] (%pa = %transport_fanout8_out4 : !fabric.bits<0> to !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%ctrl_in = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
      %value = fabric.op [@dataflow.constant] (%ctrl_in) {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002"]}]} : (!fabric.bits<0>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  %logic_shifted = fabric.pe [spatial] (%pa = %logic_shift_lhs : !fabric.bits<32>,
                    %pb = %logic_shift_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %shifted = fabric.op [@arith.shrsi, @arith.shrui] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %shifted : !fabric.bits<32>
    }
  }
  %addr_unscaled = fabric.pe [spatial] (%pa = %addr_unscale_lhs : !fabric.bits<32>,
                    %pb = %addr_unscale_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %shifted = fabric.op [@arith.shrsi, @arith.shrui] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
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
      %packed = fabric.op [@llvm.arm.qadd16, @llvm.arm.sadd16, @llvm.arm.qsub16, @llvm.arm.qsub8] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
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
  %unsigned_minmax = fabric.pe [spatial] (%pa = %transport_fanout9_out0 : !fabric.bits<32>,
                    %pb = %transport_fanout10_out0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected = fabric.op [@llvm.intr.umax] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected : !fabric.bits<32>
    }
  }
  %unsigned_min = fabric.pe [spatial] (%pa = %transport_fanout9_out1 : !fabric.bits<32>,
                    %pb = %transport_fanout10_out1 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected = fabric.op [@llvm.intr.umin] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected : !fabric.bits<32>
    }
  }
  %signed_min = fabric.pe [spatial] (%pa = %transport_fanout9_out2 : !fabric.bits<32>,
                    %pb = %transport_fanout10_out2 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected = fabric.op [@llvm.intr.smin] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected : !fabric.bits<32>
    }
  }
  %signed_max = fabric.pe [spatial] (%pa = %transport_fanout9_out3 : !fabric.bits<32>,
                    %pb = %transport_fanout10_out3 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected = fabric.op [@llvm.intr.smax] (%lhs, %rhs) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
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
  %abs = fabric.pe [spatial] (%pa = %transport_fanout11_out0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %abs = fabric.op [@llvm.intr.abs] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %abs : !fabric.bits<32>
    }
  }
  %swapped = fabric.pe [spatial] (%pa = %transport_fanout11_out1 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %swapped = fabric.op [@llvm.intr.bswap] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %swapped : !fabric.bits<32>
    }
  }
  %leading_zero_count = fabric.pe [spatial] (%pa = %transport_fanout11_out2 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
      %leading_zero_count = fabric.op [@llvm.intr.ctlz] (%value) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %leading_zero_count : !fabric.bits<32>
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
  %wide_signed_quotient = fabric.pe [spatial] (%pa = %wide_div_lhs : !fabric.bits<64>,
                    %pb = %wide_div_rhs : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %value = fabric.op [@arith.divsi] (%lhs, %rhs) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
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
  %wide_sum = fabric.pe [spatial] (%pa = %wide_add_lhs : !fabric.bits<64>,
                    %pb = %wide_add_rhs : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %value = fabric.op [@arith.addi, @arith.subi] (%lhs, %rhs) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %value : !fabric.bits<64>
    }
  }
  %wide_sum_aux = fabric.pe [spatial] (%pa = %wide_add_aux_lhs : !fabric.bits<64>,
                    %pb = %wide_add_aux_rhs : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %value = fabric.op [@arith.addi, @arith.subi] (%lhs, %rhs) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %value : !fabric.bits<64>
    }
  }
  %wide_shifted = fabric.pe [spatial] (%pa = %wide_shift_lhs : !fabric.bits<64>,
                    %pb = %wide_shift_rhs : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %value = fabric.op [@arith.shli] (%lhs, %rhs) : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
      fabric.yield %value : !fabric.bits<64>
    }
  }
  %wide_truncated_wide = fabric.pe [spatial] (%pa = %wide_trunc_input : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%value = %pa : !fabric.bits<64>) -> !fabric.bits<64> {
      %narrow = fabric.op [@llvm.trunc] (%value) : (!fabric.bits<64>) -> !fabric.bits<32>
      fabric.yield %narrow : !fabric.bits<32> to !fabric.bits<64>
    }
  }
  %wide_truncated_aux_wide = fabric.pe [spatial] (%pa = %wide_trunc_aux_input : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%value = %pa : !fabric.bits<64>) -> !fabric.bits<64> {
      %narrow = fabric.op [@llvm.trunc] (%value) : (!fabric.bits<64>) -> !fabric.bits<32>
      fabric.yield %narrow : !fabric.bits<32> to !fabric.bits<64>
    }
  }
  %fp = fabric.pe [spatial] (%pa = %transport_fanout11_out3 : !fabric.bits<32>) -> !fabric.bits<32> {
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
  %cmpf_pred = fabric.pe [spatial] (%pa = %transport_fanout12_out0 : !fabric.bits<32>,
                    %pb = %transport_fanout13_out0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %pred = fabric.op [@arith.cmpf] (%lhs, %rhs) {hw_params = [{predicate = ["oeq", "ogt", "ugt", "ule", "olt"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %cmpi_pred = fabric.pe [spatial] (%pa = %transport_fanout12_out1 : !fabric.bits<32>,
                    %pb = %transport_fanout13_out1 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %cmpi_pred_aux = fabric.pe [spatial] (%pa = %transport_fanout12_out2 : !fabric.bits<32>,
                    %pb = %transport_fanout13_out2 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pa : !fabric.bits<32>,
              %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %cmpi64_pred = fabric.pe [spatial] (%pa = %transport_fanout14_out0 : !fabric.bits<64>,
                    %pb = %transport_fanout15_out0 : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %pred = fabric.op [@arith.cmpi] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]} : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<64>
    }
  }
  %cmpi64_pred_aux = fabric.pe [spatial] (%pa = %transport_fanout14_out1 : !fabric.bits<64>,
                    %pb = %transport_fanout15_out1 : !fabric.bits<64>) -> !fabric.bits<64> {
    fabric.fu(%lhs = %pa : !fabric.bits<64>,
              %rhs = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
      %pred = fabric.op [@arith.cmpi] (%lhs, %rhs) {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]} : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<1>
      fabric.yield %pred : !fabric.bits<1> to !fabric.bits<64>
    }
  }
  %wide_pred_extui = fabric.pe [spatial] (%pa = %transport_fanout16_out0 : !fabric.bits<64>) -> !fabric.bits<64> {
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
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %when_true = %pb : !fabric.bits<32>,
              %when_false = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
      %selected_value = fabric.op [@llvm.select] (%sel, %when_true, %when_false) : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %selected_value : !fabric.bits<32>
    }
  }
  %control_demux_false, %control_demux_true = fabric.pe [spatial] (%pa = %transport_fanout17_out0 : !fabric.bits<32>,
                    %pb = %demux_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
              %value = %pb : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %false_lane, %true_lane = fabric.op [@dataflow.demux] (%sel, %value) : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %false_lane, %true_lane : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  %compute_demux_false, %compute_demux_true = fabric.pe [spatial] (%pa = %transport_fanout17_out1 : !fabric.bits<32>,
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
                    %pb = %transport_fanout8_out5 : !fabric.bits<0> to !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
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
  %vector_sync_done = fabric.pe [spatial] (%pa = %sync_head : !fabric.bits<0>,
                    %pb = %vector_sync_mid : !fabric.bits<0>,
                    %pc = %sync_tail : !fabric.bits<0>,
                    %pd = %sync_extra : !fabric.bits<0>,
                    %pe = %transport_fanout18_out0 : !fabric.bits<0>,
                    %pf = %sync_lane5 : !fabric.bits<0>,
                    %pg = %transport_fanout19_out0 : !fabric.bits<0>,
                    %ph = %sync_lane6 : !fabric.bits<0>,
                    %pi = %sync_lane7 : !fabric.bits<0>) -> !fabric.bits<0> {
    fabric.fu(%fa = %pa : !fabric.bits<0>,
              %fb = %pb : !fabric.bits<0>,
              %fc = %pc : !fabric.bits<0>,
              %fd = %pd : !fabric.bits<0>,
              %fe = %pe : !fabric.bits<0>,
              %ff = %pf : !fabric.bits<0>,
              %fg = %pg : !fabric.bits<0>,
              %fh = %ph : !fabric.bits<0>,
              %fi = %pi : !fabric.bits<0>) -> !fabric.bits<0> {
      %sync_done0, %sync_done1, %sync_done2, %sync_done3, %sync_done4, %sync_done5, %sync_done6, %sync_done7, %sync_done8 = fabric.op [@dataflow.sync] (%fa, %fb, %fc, %fd, %fe, %ff, %fg, %fh, %fi) {sw_configs = {bitmask = "111111111"}} : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %sync_done0 : !fabric.bits<0>
    }
  }
  %sync_done = fabric.pe [spatial] (%pc = %transport_fanout20_out0 : !fabric.bits<0>,
                    %pd = %sync_aux_done : !fabric.bits<0>) -> !fabric.bits<0> {
    fabric.fu(%fc = %pc : !fabric.bits<0>,
              %fd = %pd : !fabric.bits<0>) -> !fabric.bits<0> {
      %sync_done0, %sync_done1 = fabric.op [@dataflow.sync] (%fc, %fd) {sw_configs = {bitmask = "11"}} : (!fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %sync_done0 : !fabric.bits<0>
    }
  }
  %typed_sync_i1_done_wide, %typed_sync_i1_published = fabric.pe [spatial] (%pc = %typed_sync_i1_control : !fabric.bits<0> to !fabric.bits<32>,
                    %pv = %typed_sync_i1_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%control = %pc : !fabric.bits<32> to !fabric.bits<0>,
              %value = %pv : !fabric.bits<32> to !fabric.bits<1>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %done, %published = fabric.op [@dataflow.sync] (%control, %value) {sw_configs = {bitmask = "11"}} : (!fabric.bits<0>, !fabric.bits<1>) -> (!fabric.bits<0>, !fabric.bits<1>)
      fabric.yield %done : !fabric.bits<0> to !fabric.bits<32>, %published : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %typed_sync_i8_done_wide, %typed_sync_i8_published = fabric.pe [spatial] (%pc = %typed_sync_i8_control : !fabric.bits<0> to !fabric.bits<32>,
                    %pv = %typed_sync_i8_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%control = %pc : !fabric.bits<32> to !fabric.bits<0>,
              %value = %pv : !fabric.bits<32> to !fabric.bits<8>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %done, %published = fabric.op [@dataflow.sync] (%control, %value) {sw_configs = {bitmask = "11"}} : (!fabric.bits<0>, !fabric.bits<8>) -> (!fabric.bits<0>, !fabric.bits<8>)
      fabric.yield %done : !fabric.bits<0> to !fabric.bits<32>, %published : !fabric.bits<8> to !fabric.bits<32>
    }
  }
  %typed_sync_i32_done_wide, %typed_sync_i32_published = fabric.pe [spatial] (%pc = %typed_sync_i32_control : !fabric.bits<0> to !fabric.bits<32>,
                    %pv = %typed_sync_i32_value : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%control = %pc : !fabric.bits<32> to !fabric.bits<0>,
              %value = %pv : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %done, %published = fabric.op [@dataflow.sync] (%control, %value) {sw_configs = {bitmask = "11"}} : (!fabric.bits<0>, !fabric.bits<32>) -> (!fabric.bits<0>, !fabric.bits<32>)
      fabric.yield %done : !fabric.bits<0> to !fabric.bits<32>, %published : !fabric.bits<32>
    }
  }
  %typed_sync_i64_done_wide, %typed_sync_i64_published = fabric.pe [spatial] (%pc = %typed_sync_i64_control : !fabric.bits<0> to !fabric.bits<64>,
                    %pv = %typed_sync_i64_value : !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.fu(%control = %pc : !fabric.bits<64> to !fabric.bits<0>,
              %value = %pv : !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
      %done, %published = fabric.op [@dataflow.sync] (%control, %value) {sw_configs = {bitmask = "11"}} : (!fabric.bits<0>, !fabric.bits<64>) -> (!fabric.bits<0>, !fabric.bits<64>)
      fabric.yield %done : !fabric.bits<0> to !fabric.bits<64>, %published : !fabric.bits<64>
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
  %wide_index_cast0 = fabric.pe [spatial] (%value = %wide_index_cast0_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_cast1 = fabric.pe [spatial] (%value = %wide_index_cast1_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_truncated = fabric.fifo %wide_truncated_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_truncated_aux = fabric.fifo %wide_truncated_aux_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast0_narrow = fabric.fifo %wide_index_cast0 [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast1_narrow = fabric.fifo %wide_index_cast1 [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %cmpi64_pred_aux_narrow = fabric.fifo %cmpi64_pred_aux [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %typed_sync_i1_done = fabric.fifo %typed_sync_i1_done_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %typed_sync_i8_done = fabric.fifo %typed_sync_i8_done_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %typed_sync_i32_done = fabric.fifo %typed_sync_i32_done_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %typed_sync_i64_done = fabric.fifo %typed_sync_i64_done_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<0>
  %logic_mask_lhs = fabric.switch [spatial] %transport_fanout11_out4, %transport_fanout4_out1, %transport_fanout21_out0, %transport_fanout22_out0, %transport_fanout23_out0, %transport_fanout24_out0, %transport_fanout25_out0, %transport_fanout26_out0, %transport_fanout27_out0, %transport_fanout28_out0, %transport_fanout29_out0
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %logic_mask_rhs = fabric.switch [spatial] %transport_fanout30_out0, %transport_fanout31_out0, %transport_fanout32_out0, %transport_fanout33_out0, %transport_fanout34_out0, %transport_fanout35_out0, %transport_fanout36_out0, %transport_fanout27_out1, %transport_fanout28_out1
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_mask_lhs = fabric.switch [spatial] %transport_fanout37_out0, %transport_fanout22_out1, %transport_fanout38_out0, %transport_fanout39_out0, %transport_fanout40_out0, %transport_fanout29_out1, %transport_fanout41_out0, %transport_fanout4_out2, %transport_fanout21_out1
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_mask_rhs = fabric.switch [spatial] %transport_fanout11_out5, %transport_fanout30_out1, %transport_fanout31_out1, %transport_fanout42_out0, %transport_fanout32_out1, %transport_fanout33_out1, %transport_fanout34_out1, %transport_fanout35_out1, %transport_fanout36_out1, %transport_fanout43_out0, %transport_fanout44_out0, %transport_fanout45_out0
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_xor_lhs = fabric.switch [spatial] %transport_fanout39_out1, %transport_fanout40_out1, %transport_fanout23_out1, %transport_fanout24_out1, %transport_fanout37_out1, %transport_fanout22_out2, %transport_fanout38_out1, %transport_fanout46_out0, %transport_fanout47_out0, %transport_fanout25_out1, %transport_fanout48_out0
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_xor_rhs = fabric.switch [spatial] %transport_fanout37_out2, %transport_fanout22_out3, %transport_fanout38_out2, %transport_fanout11_out6, %transport_fanout30_out2, %transport_fanout4_out3, %transport_fanout21_out2, %transport_fanout31_out2, %transport_fanout42_out1, %transport_fanout32_out2, %transport_fanout33_out2, %transport_fanout34_out2, %transport_fanout35_out2, %transport_fanout36_out2
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_add_lhs = fabric.switch [spatial] %transport_fanout11_out7, %transport_fanout21_out3, %transport_fanout4_out4, %transport_fanout37_out3, %transport_fanout29_out2, %transport_fanout49_out0, %transport_fanout22_out4, %transport_fanout32_out3, %transport_fanout50_out0, %transport_fanout51_out0, %transport_fanout33_out3, %transport_fanout34_out3, %transport_fanout35_out3, %transport_fanout36_out3, %transport_fanout52_out0, %transport_fanout53_out0, %transport_fanout54_out0, %transport_fanout55_out0, %transport_fanout56_out0, %transport_fanout57_out0, %transport_fanout58_out0, %transport_fanout59_out0, %transport_fanout60_out0, %transport_fanout61_out0
        [{connectivity_table = ["111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_add_rhs = fabric.switch [spatial] %transport_fanout30_out3, %transport_fanout4_out5, %transport_fanout21_out4, %transport_fanout33_out4, %transport_fanout41_out1, %transport_fanout32_out4, %transport_fanout34_out4, %transport_fanout35_out4, %transport_fanout36_out4, %transport_fanout62_out0, %transport_fanout63_out0, %transport_fanout64_out0, %transport_fanout55_out1, %transport_fanout56_out1, %transport_fanout57_out1, %transport_fanout58_out1, %transport_fanout59_out1, %transport_fanout50_out1, %transport_fanout51_out1, %transport_fanout49_out1
        [{connectivity_table = ["11111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_mul_lhs = fabric.switch [spatial] %transport_fanout11_out8, %transport_fanout25_out2, %transport_fanout4_out6, %transport_fanout21_out5, %transport_fanout65_out0, %transport_fanout66_out0, %transport_fanout63_out1, %transport_fanout64_out1, %transport_fanout34_out5, %transport_fanout35_out5, %transport_fanout36_out5, %transport_fanout55_out2, %transport_fanout56_out2, %transport_fanout57_out2, %transport_fanout58_out2, %transport_fanout67_out0, %transport_fanout29_out3
        [{connectivity_table = ["11111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_mul_rhs = fabric.switch [spatial] %transport_fanout30_out4, %transport_fanout4_out7, %transport_fanout21_out6, %transport_fanout32_out5, %transport_fanout34_out6, %transport_fanout35_out6, %transport_fanout36_out6, %transport_fanout52_out1, %transport_fanout53_out1, %transport_fanout54_out1, %transport_fanout33_out5, %transport_fanout55_out3, %transport_fanout56_out3, %transport_fanout57_out3, %transport_fanout58_out3
        [{connectivity_table = ["111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_mul_aux_lhs = fabric.switch [spatial] %transport_fanout11_out9, %transport_fanout25_out3, %transport_fanout4_out8, %transport_fanout21_out7, %transport_fanout65_out1, %transport_fanout66_out1, %transport_fanout63_out2, %transport_fanout64_out2
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_mul_aux_rhs = fabric.switch [spatial] %transport_fanout30_out5, %transport_fanout4_out9, %transport_fanout21_out8, %transport_fanout32_out6, %transport_fanout34_out7, %transport_fanout35_out7, %transport_fanout36_out7, %transport_fanout52_out2, %transport_fanout53_out2, %transport_fanout54_out2, %transport_fanout33_out6, %aux_active_idx1
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_div0_lhs = fabric.switch [spatial] %transport_fanout67_out1, %transport_fanout68_out0, %transport_fanout63_out3, %transport_fanout64_out3, %transport_fanout30_out6, %transport_fanout31_out3
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_div0_rhs = fabric.switch [spatial] %transport_fanout31_out4, %transport_fanout32_out7, %transport_fanout33_out7, %transport_fanout34_out8, %transport_fanout35_out8, %transport_fanout36_out8, %transport_fanout52_out3, %transport_fanout53_out3, %transport_fanout54_out3
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_div1_lhs = fabric.switch [spatial] %transport_fanout67_out2, %transport_fanout68_out1, %transport_fanout63_out4, %transport_fanout64_out4, %transport_fanout30_out7, %transport_fanout31_out5
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_div1_rhs = fabric.switch [spatial] %transport_fanout31_out6, %transport_fanout32_out8, %transport_fanout33_out8, %transport_fanout34_out9, %transport_fanout35_out9, %transport_fanout36_out9, %transport_fanout52_out4, %transport_fanout53_out4, %transport_fanout54_out4, %transport_fanout64_out5
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_rem_lhs = fabric.switch [spatial] %transport_fanout63_out5, %transport_fanout64_out6, %transport_fanout11_out10, %transport_fanout30_out8
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_rem_rhs = fabric.switch [spatial] %transport_fanout32_out9, %transport_fanout34_out10, %transport_fanout35_out10, %transport_fanout36_out10, %transport_fanout52_out5, %transport_fanout53_out5, %transport_fanout54_out5, %transport_fanout42_out2, %transport_fanout64_out7
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %uint_rem_lhs = fabric.switch [spatial] %transport_fanout50_out2, %transport_fanout63_out6, %transport_fanout64_out8, %transport_fanout30_out9, %transport_fanout40_out2, %transport_fanout29_out4
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %uint_rem_rhs = fabric.switch [spatial] %transport_fanout31_out7, %transport_fanout32_out10, %transport_fanout34_out11, %transport_fanout35_out11, %transport_fanout36_out11, %transport_fanout52_out6, %transport_fanout53_out6, %transport_fanout54_out6
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_or_lhs = fabric.switch [spatial] %transport_fanout11_out11, %transport_fanout46_out1, %transport_fanout4_out10, %transport_fanout21_out9, %transport_fanout40_out3, %transport_fanout39_out2
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_or_rhs = fabric.switch [spatial] %transport_fanout30_out10, %transport_fanout46_out2, %transport_fanout4_out11, %transport_fanout21_out10
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_xor_lhs = fabric.switch [spatial] %transport_fanout11_out12, %transport_fanout69_out0, %transport_fanout24_out2, %transport_fanout23_out2, %transport_fanout46_out3, %transport_fanout4_out12, %transport_fanout70_out0, %transport_fanout39_out3, %transport_fanout47_out1, %transport_fanout48_out1, %transport_fanout71_out0, %transport_fanout27_out2, %transport_fanout28_out2
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_xor_rhs = fabric.switch [spatial] %transport_fanout30_out11, %transport_fanout21_out11, %transport_fanout4_out13, %transport_fanout46_out4, %transport_fanout32_out11, %transport_fanout33_out9, %transport_fanout34_out12, %transport_fanout35_out12, %transport_fanout36_out12, %transport_fanout37_out4, %transport_fanout22_out5, %transport_fanout38_out3, %transport_fanout47_out2, %transport_fanout39_out4, %transport_fanout48_out2
        [{connectivity_table = ["111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %packed_sat_lhs = fabric.switch [spatial] %transport_fanout11_out13, %transport_fanout32_out12, %transport_fanout33_out10, %transport_fanout34_out13, %transport_fanout35_out13, %transport_fanout36_out13, %transport_fanout55_out4, %transport_fanout56_out4, %transport_fanout57_out4, %transport_fanout58_out4
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %packed_sat_rhs = fabric.switch [spatial] %transport_fanout46_out5, %transport_fanout47_out3, %transport_fanout4_out14, %transport_fanout21_out12, %transport_fanout30_out12, %transport_fanout32_out13, %transport_fanout33_out11, %transport_fanout34_out14, %transport_fanout35_out14, %transport_fanout36_out14, %transport_fanout55_out5, %transport_fanout56_out5, %transport_fanout57_out5, %transport_fanout58_out5
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %minmax_lhs = fabric.switch [spatial] %transport_fanout11_out14, %transport_fanout30_out13, %transport_fanout4_out15, %transport_fanout21_out13, %transport_fanout41_out2, %transport_fanout29_out5, %transport_fanout67_out3, %transport_fanout68_out2, %transport_fanout47_out4, %transport_fanout46_out6, %transport_fanout37_out5, %transport_fanout22_out6, %transport_fanout38_out4, %transport_fanout55_out6, %transport_fanout56_out6, %transport_fanout57_out6, %transport_fanout58_out6
        [{connectivity_table = ["11111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %minmax_rhs = fabric.switch [spatial] %transport_fanout30_out14, %transport_fanout31_out8, %transport_fanout4_out16, %transport_fanout21_out14, %transport_fanout41_out3, %transport_fanout29_out6, %transport_fanout67_out4, %transport_fanout68_out3, %transport_fanout43_out1, %transport_fanout44_out1, %transport_fanout45_out1, %transport_fanout32_out14, %transport_fanout33_out12, %transport_fanout34_out15, %transport_fanout35_out15, %transport_fanout36_out15
        [{connectivity_table = ["1111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %rotate_lhs = fabric.switch [spatial] %transport_fanout11_out15, %transport_fanout21_out15, %transport_fanout4_out17, %transport_fanout46_out7, %transport_fanout67_out5, %transport_fanout50_out3
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %rotate_rhs = fabric.switch [spatial] %transport_fanout30_out15, %transport_fanout21_out16, %transport_fanout4_out18, %transport_fanout46_out8, %transport_fanout67_out6, %transport_fanout50_out4
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %rotate_amount = fabric.switch [spatial] %transport_fanout31_out9, %transport_fanout4_out19, %transport_fanout32_out15, %transport_fanout43_out2
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %cmp_lhs = fabric.switch [spatial] %transport_fanout11_out16, %transport_fanout46_out9, %transport_fanout4_out20, %transport_fanout21_out17, %transport_fanout22_out7, %transport_fanout29_out7, %transport_fanout68_out4, %transport_fanout47_out5, %transport_fanout48_out3
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %cmp_rhs = fabric.switch [spatial] %transport_fanout30_out16, %transport_fanout31_out10, %transport_fanout32_out16, %transport_fanout21_out18, %transport_fanout4_out21, %transport_fanout33_out13, %transport_fanout34_out16, %transport_fanout35_out16, %transport_fanout36_out16, %transport_fanout48_out4
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %int_extui_input = fabric.switch [spatial] %transport_fanout11_out17, %transport_fanout27_out3, %transport_fanout28_out3, %transport_fanout71_out1, %transport_fanout72_out0, %transport_fanout46_out10, %transport_fanout47_out6, %transport_fanout25_out4, %transport_fanout26_out1
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %select_pred = fabric.switch [spatial] %transport_fanout11_out18, %transport_fanout46_out11, %transport_fanout47_out7, %transport_fanout27_out4, %transport_fanout28_out4, %transport_fanout72_out1, %transport_fanout71_out2, %transport_fanout48_out5
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %select_true = fabric.switch [spatial] %transport_fanout30_out17, %transport_fanout41_out4, %transport_fanout21_out19, %transport_fanout69_out1, %transport_fanout4_out22, %transport_fanout67_out7, %transport_fanout68_out5, %transport_fanout40_out4, %transport_fanout32_out17, %transport_fanout34_out17, %transport_fanout35_out17, %transport_fanout36_out17, %transport_fanout55_out7, %transport_fanout56_out7, %transport_fanout57_out7, %transport_fanout58_out7, %transport_fanout26_out2, %transport_fanout37_out6
        [{connectivity_table = ["111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %select_false = fabric.switch [spatial] %transport_fanout31_out11, %transport_fanout69_out2, %transport_fanout4_out23, %transport_fanout21_out20, %transport_fanout37_out7, %transport_fanout22_out8, %transport_fanout43_out3, %transport_fanout44_out2, %transport_fanout45_out2, %transport_fanout26_out3, %transport_fanout29_out8, %transport_fanout40_out5
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %gate_cond = fabric.switch [spatial] %transport_fanout73_out0, %transport_fanout46_out12, %transport_fanout47_out8, %transport_fanout27_out5, %transport_fanout28_out5, %transport_fanout72_out2, %transport_fanout71_out3, %transport_fanout5_out2, %transport_fanout11_out19
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %gate_value = fabric.switch [spatial] %transport_fanout63_out7, %transport_fanout41_out5, %transport_fanout29_out9, %transport_fanout68_out6, %transport_fanout67_out8, %transport_fanout49_out2, %transport_fanout37_out8, %transport_fanout55_out8, %transport_fanout56_out8, %transport_fanout57_out8, %transport_fanout58_out8, %transport_fanout34_out18
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %gate_value1 = fabric.switch [spatial] %transport_fanout63_out8, %transport_fanout41_out6, %transport_fanout29_out10, %transport_fanout68_out7, %transport_fanout67_out9, %transport_fanout49_out3, %transport_fanout37_out9, %transport_fanout55_out9, %transport_fanout56_out9, %transport_fanout57_out9, %transport_fanout58_out9, %transport_fanout35_out18
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %demux_sel = fabric.switch [spatial] %transport_fanout46_out13, %transport_fanout47_out9, %transport_fanout27_out6, %transport_fanout28_out6, %transport_fanout72_out3, %transport_fanout71_out4, %transport_fanout5_out3, %transport_fanout11_out20
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %demux_value = fabric.switch [spatial] %transport_fanout37_out10, %transport_fanout22_out9, %transport_fanout38_out5, %transport_fanout33_out14, %transport_fanout32_out18, %transport_fanout29_out11
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %demux_then_value = fabric.switch [spatial] %transport_fanout74_out0, %transport_fanout75_out0, %transport_fanout76_out0, %transport_fanout77_out0, %transport_fanout78_out0, %transport_fanout79_out0, %transport_fanout80_out0, %transport_fanout4_out24, %transport_fanout21_out21, %transport_fanout67_out10, %transport_fanout68_out8, %transport_fanout50_out5, %transport_fanout51_out2, %transport_fanout39_out5
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mux_sel = fabric.switch [spatial] %transport_fanout46_out14, %transport_fanout47_out10, %transport_fanout27_out7, %transport_fanout28_out7, %transport_fanout72_out4, %transport_fanout71_out5, %transport_fanout5_out4, %transport_fanout11_out21
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mux_false = fabric.switch [spatial] %transport_fanout81_out0, %transport_fanout37_out11, %transport_fanout22_out10, %transport_fanout38_out6, %transport_fanout33_out15
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mux_true = fabric.switch [spatial] %transport_fanout82_out0, %transport_fanout74_out1, %transport_fanout75_out1, %transport_fanout76_out1, %transport_fanout77_out1, %transport_fanout78_out1, %transport_fanout79_out1, %transport_fanout80_out1, %transport_fanout4_out25, %transport_fanout21_out22
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %control_token_demux_sel = fabric.switch [spatial] %transport_fanout46_out15, %transport_fanout47_out11, %transport_fanout27_out8, %transport_fanout28_out8, %transport_fanout72_out5, %transport_fanout71_out6, %transport_fanout5_out5, %transport_fanout11_out22
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %control_token_demux_false_token = fabric.fifo %control_token_demux_false [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %control_token_demux_true_token = fabric.fifo %control_token_demux_true [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %control_token_muxed_token = fabric.fifo %control_token_muxed [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<0>
  %control_token_mux_sel = fabric.switch [spatial] %transport_fanout46_out16, %transport_fanout47_out12, %transport_fanout27_out9, %transport_fanout28_out9, %transport_fanout72_out6, %transport_fanout71_out7, %transport_fanout5_out6, %transport_fanout11_out23
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %control_token_mux_false = fabric.switch [spatial] %transport_fanout83_out0, %transport_fanout19_out1, %transport_fanout8_out6
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %control_token_mux_true = fabric.switch [spatial] %transport_fanout19_out2, %transport_fanout84_out0, %transport_fanout8_out7
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %load1_addr = fabric.switch [spatial] %transport_fanout41_out7, %transport_fanout30_out18, %transport_fanout23_out3, %transport_fanout55_out10, %transport_fanout56_out10, %transport_fanout57_out10, %transport_fanout58_out10, %transport_fanout29_out12, %transport_fanout68_out9, %transport_fanout49_out4, %transport_fanout67_out11, %transport_fanout37_out12, %transport_fanout63_out9, %transport_fanout64_out9, %transport_fanout39_out6, %transport_fanout46_out17, %transport_fanout47_out13, %transport_fanout59_out2, %transport_fanout43_out4, %transport_fanout44_out3, %transport_fanout45_out3, %transport_fanout85_out0, %transport_fanout86_out0, %transport_fanout87_out0, %transport_fanout88_out0
        [{connectivity_table = ["1111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %cast0_input = fabric.switch [spatial] %transport_fanout11_out24, %transport_fanout4_out26, %transport_fanout21_out23, %transport_fanout46_out18, %transport_fanout70_out1, %transport_fanout41_out8, %transport_fanout29_out13, %transport_fanout67_out12, %transport_fanout68_out10, %transport_fanout89_out0
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %cast1_input = fabric.switch [spatial] %transport_fanout11_out25, %transport_fanout4_out27, %transport_fanout21_out24, %transport_fanout46_out19, %transport_fanout70_out2, %transport_fanout41_out9, %transport_fanout29_out14, %transport_fanout67_out13, %transport_fanout68_out11, %transport_fanout89_out1, %transport_fanout55_out11
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %cast2_input = fabric.switch [spatial] %transport_fanout11_out26, %transport_fanout4_out28, %transport_fanout21_out25, %transport_fanout46_out20, %transport_fanout70_out3, %transport_fanout41_out10, %transport_fanout29_out15, %transport_fanout67_out14, %transport_fanout68_out12, %transport_fanout89_out2, %transport_fanout55_out12, %transport_fanout56_out11
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %cast3_input = fabric.switch [spatial] %transport_fanout11_out27, %transport_fanout4_out29, %transport_fanout21_out26, %transport_fanout46_out21, %transport_fanout70_out4, %transport_fanout41_out11, %transport_fanout29_out16, %transport_fanout67_out15, %transport_fanout68_out13, %transport_fanout89_out3, %transport_fanout55_out13, %transport_fanout56_out12, %transport_fanout57_out11
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %wide_zext0_input = fabric.switch [spatial] %transport_fanout4_out30, %transport_fanout21_out27, %transport_fanout11_out28, %transport_fanout55_out14, %transport_fanout56_out13, %unsigned_minmax, %unsigned_min, %transport_fanout90_out0, %transport_fanout91_out0
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %wide_zext1_input = fabric.switch [spatial] %transport_fanout21_out28, %transport_fanout4_out31, %transport_fanout30_out19, %transport_fanout55_out15, %transport_fanout56_out14
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %wide_mul_lhs = fabric.switch [spatial] %transport_fanout92_out0, %transport_fanout93_out0, %transport_fanout0_out1, %transport_fanout1_out1
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_mul_rhs = fabric.switch [spatial] %transport_fanout93_out1, %transport_fanout92_out1, %transport_fanout0_out2, %transport_fanout2_out1
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_div_lhs = fabric.switch [spatial] %transport_fanout94_out0, %transport_fanout93_out2, %transport_fanout92_out2, %transport_fanout0_out3
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_div_rhs = fabric.switch [spatial] %transport_fanout0_out4, %transport_fanout1_out2, %transport_fanout2_out2, %transport_fanout93_out3, %transport_fanout92_out3
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_rem_lhs = fabric.switch [spatial] %transport_fanout94_out1, %transport_fanout93_out4, %transport_fanout92_out4, %transport_fanout0_out5
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_rem_rhs = fabric.switch [spatial] %transport_fanout0_out6, %transport_fanout1_out3, %transport_fanout2_out3, %transport_fanout93_out5, %transport_fanout92_out5
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_add_lhs = fabric.switch [spatial] %transport_fanout0_out7, %transport_fanout1_out4, %transport_fanout2_out4, %transport_fanout95_out0, %transport_fanout93_out6, %transport_fanout92_out6, %transport_fanout94_out2, %transport_fanout96_out0, %transport_fanout97_out0
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_add_rhs = fabric.switch [spatial] %transport_fanout0_out8, %transport_fanout1_out5, %transport_fanout2_out5, %transport_fanout95_out1, %transport_fanout93_out7, %transport_fanout92_out7, %transport_fanout94_out3, %transport_fanout96_out1, %transport_fanout97_out1
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_add_aux_lhs = fabric.switch [spatial] %transport_fanout0_out9, %transport_fanout1_out6, %transport_fanout2_out6, %transport_fanout95_out2, %transport_fanout98_out0, %transport_fanout93_out8, %transport_fanout92_out8, %transport_fanout94_out4, %transport_fanout96_out2, %transport_fanout97_out2
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_add_aux_rhs = fabric.switch [spatial] %transport_fanout0_out10, %transport_fanout1_out7, %transport_fanout2_out7, %transport_fanout95_out3, %transport_fanout98_out1, %transport_fanout93_out9, %transport_fanout92_out9, %transport_fanout94_out5, %transport_fanout96_out3, %transport_fanout97_out3
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_shift_lhs = fabric.switch [spatial] %transport_fanout0_out11, %transport_fanout1_out8, %transport_fanout2_out8, %transport_fanout98_out2, %transport_fanout99_out0, %transport_fanout93_out10, %transport_fanout92_out10, %transport_fanout94_out6, %transport_fanout96_out4, %transport_fanout97_out4
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_shift_rhs = fabric.switch [spatial] %transport_fanout0_out12, %transport_fanout1_out9, %transport_fanout2_out9, %transport_fanout93_out11, %transport_fanout92_out11
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_trunc_input = fabric.switch [spatial] %transport_fanout97_out5, %transport_fanout94_out7, %transport_fanout93_out12, %transport_fanout92_out12, %transport_fanout100_out0, %transport_fanout96_out5, %transport_fanout95_out4, %transport_fanout98_out3, %transport_fanout99_out1
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_trunc_aux_input = fabric.switch [spatial] %transport_fanout98_out4, %transport_fanout99_out2, %transport_fanout95_out5, %transport_fanout97_out6, %transport_fanout94_out8, %transport_fanout96_out6, %transport_fanout93_out13, %transport_fanout92_out13, %transport_fanout100_out1
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_index_cast0_input = fabric.switch [spatial] %transport_fanout0_out13, %transport_fanout1_out10, %transport_fanout2_out10, %transport_fanout93_out14, %transport_fanout92_out14, %transport_fanout94_out9, %transport_fanout98_out5, %transport_fanout99_out3, %transport_fanout95_out6, %transport_fanout96_out7, %transport_fanout97_out7
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %wide_index_cast1_input = fabric.switch [spatial] %transport_fanout0_out14, %transport_fanout1_out11, %transport_fanout2_out11, %transport_fanout93_out15, %transport_fanout92_out15, %transport_fanout94_out10, %transport_fanout98_out6, %transport_fanout99_out4, %transport_fanout95_out7, %transport_fanout96_out8, %transport_fanout97_out8
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %cmp64_lhs = fabric.switch [spatial] %transport_fanout0_out15, %transport_fanout1_out12, %transport_fanout2_out12, %transport_fanout93_out16, %transport_fanout92_out16, %transport_fanout94_out11, %transport_fanout96_out9, %transport_fanout97_out9, %transport_fanout95_out8
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %cmp64_rhs = fabric.switch [spatial] %transport_fanout0_out16, %transport_fanout1_out13, %transport_fanout2_out13, %transport_fanout93_out17, %transport_fanout92_out17, %transport_fanout94_out12, %transport_fanout96_out10, %transport_fanout97_out10, %transport_fanout95_out9
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %fp_negated_input = fabric.switch [spatial] %transport_fanout4_out32, %transport_fanout21_out29, %transport_fanout101_out0, %transport_fanout102_out0, %transport_fanout103_out0, %transport_fanout104_out0, %transport_fanout78_out2, %transport_fanout79_out2, %transport_fanout105_out0, %transport_fanout106_out0, %transport_fanout80_out2
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %load2_addr = fabric.switch [spatial] %transport_fanout31_out12, %transport_fanout55_out16, %transport_fanout56_out15, %transport_fanout57_out12, %transport_fanout58_out11, %transport_fanout41_out12, %transport_fanout68_out14, %transport_fanout29_out17, %transport_fanout49_out5, %transport_fanout67_out16, %transport_fanout63_out10, %transport_fanout64_out10, %transport_fanout4_out33, %transport_fanout21_out30, %transport_fanout59_out3, %transport_fanout43_out5, %transport_fanout44_out4, %transport_fanout45_out4, %transport_fanout85_out1, %transport_fanout86_out1, %transport_fanout87_out1, %transport_fanout88_out1
        [{connectivity_table = ["1111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %store0_value = fabric.switch [spatial] %scan_store_value, %transport_fanout78_out3, %transport_fanout79_out3, %transport_fanout29_out18, %transport_fanout74_out2, %transport_fanout75_out2, %transport_fanout76_out2, %transport_fanout77_out2, %transport_fanout4_out34, %transport_fanout21_out31, %transport_fanout101_out1, %transport_fanout102_out1, %transport_fanout103_out1, %transport_fanout104_out1, %transport_fanout39_out7, %transport_fanout69_out3, %transport_fanout47_out14, %transport_fanout46_out22, %transport_fanout107_out0, %transport_fanout25_out5, %transport_fanout70_out5, %transport_fanout55_out17, %transport_fanout56_out16, %transport_fanout57_out13, %transport_fanout58_out12, %transport_fanout108_out0, %transport_fanout80_out3, %transport_fanout109_out0, %transport_fanout50_out6, %transport_fanout32_out19, %transport_fanout67_out17, %transport_fanout68_out15, %transport_fanout105_out1, %transport_fanout106_out1, %transport_fanout110_out0, %transport_fanout82_out1, %transport_fanout60_out1, %transport_fanout111_out0, %transport_fanout90_out1, %transport_fanout91_out1
        [{connectivity_table = ["1111111111111111111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %store1_value = fabric.switch [spatial] %transport_fanout42_out3, %transport_fanout4_out35, %transport_fanout21_out32, %transport_fanout101_out2, %transport_fanout102_out2, %transport_fanout103_out2, %transport_fanout104_out2, %transport_fanout39_out8, %transport_fanout80_out4, %transport_fanout109_out1, %transport_fanout74_out3, %transport_fanout75_out3, %transport_fanout76_out3, %transport_fanout77_out3, %transport_fanout90_out2, %transport_fanout91_out2
        [{connectivity_table = ["1111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %vector_sync_mid = fabric.switch [spatial] %transport_fanout20_out1, %transport_fanout112_out0, %transport_fanout19_out3, %transport_fanout83_out1, %transport_fanout113_out0
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %sync_head = fabric.switch [spatial] %transport_fanout20_out2, %transport_fanout19_out4, %transport_fanout83_out2
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %sync_tail = fabric.switch [spatial] %transport_fanout19_out5, %transport_fanout114_out0
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %sync_extra = fabric.switch [spatial] %transport_fanout115_out0, %transport_fanout116_out0, %transport_fanout19_out6
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %sync_lane5 = fabric.switch [spatial] %transport_fanout117_out0, %transport_fanout19_out7
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %sync_lane6 = fabric.switch [spatial] %transport_fanout112_out1, %transport_fanout18_out1, %transport_fanout19_out8
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %sync_lane7 = fabric.switch [spatial] %transport_fanout114_out1, %transport_fanout117_out1, %transport_fanout113_out1
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %typed_sync_i1_control = fabric.switch [spatial] %transport_fanout8_out8, %transport_fanout20_out3, %transport_fanout112_out2, %transport_fanout114_out2, %transport_fanout116_out1, %transport_fanout18_out2, %transport_fanout117_out2, %transport_fanout19_out9, %transport_fanout115_out1, %transport_fanout118_out0, %transport_fanout119_out0, %transport_fanout83_out3, %transport_fanout84_out1, %transport_fanout113_out2
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %typed_sync_i8_control = fabric.switch [spatial] %transport_fanout8_out9, %transport_fanout20_out4, %transport_fanout112_out3, %transport_fanout114_out3, %transport_fanout116_out2, %transport_fanout18_out3, %transport_fanout117_out3, %transport_fanout19_out10, %transport_fanout115_out2, %transport_fanout118_out1, %transport_fanout119_out1, %transport_fanout83_out4, %transport_fanout84_out2, %transport_fanout113_out3
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %typed_sync_i32_control = fabric.switch [spatial] %transport_fanout8_out10, %transport_fanout20_out5, %transport_fanout112_out4, %transport_fanout114_out4, %transport_fanout116_out3, %transport_fanout18_out4, %transport_fanout117_out4, %transport_fanout19_out11, %transport_fanout115_out3, %transport_fanout118_out2, %transport_fanout119_out2, %transport_fanout83_out5, %transport_fanout84_out3, %transport_fanout113_out4
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %typed_sync_i64_control = fabric.switch [spatial] %transport_fanout8_out11, %transport_fanout20_out6, %transport_fanout112_out5, %transport_fanout114_out5, %transport_fanout116_out4, %transport_fanout18_out5, %transport_fanout117_out5, %transport_fanout19_out12, %transport_fanout115_out4, %transport_fanout118_out3, %transport_fanout119_out3, %transport_fanout83_out6, %transport_fanout84_out4, %transport_fanout113_out5
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %typed_sync_i1_value = fabric.switch [spatial] %transport_fanout11_out29, %transport_fanout30_out20, %transport_fanout31_out13, %transport_fanout42_out4, %transport_fanout41_out13, %transport_fanout63_out11, %transport_fanout29_out19, %transport_fanout37_out13, %transport_fanout22_out11, %transport_fanout38_out7, %transport_fanout4_out36, %transport_fanout21_out33, %transport_fanout101_out3, %transport_fanout102_out3, %transport_fanout103_out3, %transport_fanout104_out3, %transport_fanout67_out18, %transport_fanout68_out16, %transport_fanout50_out7, %transport_fanout51_out3, %transport_fanout65_out2, %transport_fanout66_out2, %transport_fanout62_out1, %transport_fanout89_out4, %transport_fanout107_out1, %transport_fanout25_out6, %transport_fanout26_out4, %transport_fanout24_out3, %transport_fanout40_out6, %transport_fanout46_out23, %transport_fanout47_out15, %transport_fanout48_out6, %transport_fanout39_out9, %transport_fanout69_out4, %transport_fanout70_out6, %transport_fanout120_out0, %transport_fanout55_out18, %transport_fanout56_out17, %transport_fanout57_out14, %transport_fanout58_out13, %transport_fanout59_out4, %transport_fanout78_out4, %transport_fanout79_out4, %transport_fanout105_out2, %transport_fanout106_out2, %transport_fanout111_out1, %transport_fanout80_out5, %transport_fanout109_out2, %transport_fanout81_out1, %transport_fanout121_out0, %transport_fanout110_out1, %transport_fanout82_out2, %transport_fanout27_out10, %transport_fanout28_out10, %transport_fanout71_out8
        [{connectivity_table = ["1111111111111111111111111111111111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %typed_sync_i8_value = fabric.switch [spatial] %transport_fanout11_out30, %transport_fanout30_out21, %transport_fanout31_out14, %transport_fanout42_out5, %transport_fanout41_out14, %transport_fanout63_out12, %transport_fanout29_out20, %transport_fanout37_out14, %transport_fanout22_out12, %transport_fanout38_out8, %transport_fanout4_out37, %transport_fanout21_out34, %transport_fanout101_out4, %transport_fanout102_out4, %transport_fanout103_out4, %transport_fanout104_out4, %transport_fanout67_out19, %transport_fanout68_out17, %transport_fanout50_out8, %transport_fanout51_out4, %transport_fanout65_out3, %transport_fanout66_out3, %transport_fanout62_out2, %transport_fanout89_out5, %transport_fanout107_out2, %transport_fanout25_out7, %transport_fanout26_out5, %transport_fanout24_out4, %transport_fanout40_out7, %transport_fanout46_out24, %transport_fanout47_out16, %transport_fanout48_out7, %transport_fanout39_out10, %transport_fanout69_out5, %transport_fanout70_out7, %transport_fanout120_out1, %transport_fanout55_out19, %transport_fanout56_out18, %transport_fanout57_out15, %transport_fanout58_out14, %transport_fanout59_out5, %transport_fanout78_out5, %transport_fanout79_out5, %transport_fanout105_out3, %transport_fanout106_out3, %transport_fanout111_out2, %transport_fanout80_out6, %transport_fanout109_out3, %transport_fanout81_out2, %transport_fanout121_out1, %transport_fanout110_out2, %transport_fanout82_out3, %transport_fanout27_out11, %transport_fanout28_out11, %transport_fanout71_out9
        [{connectivity_table = ["1111111111111111111111111111111111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %typed_sync_i32_value = fabric.switch [spatial] %transport_fanout11_out31, %transport_fanout30_out22, %transport_fanout31_out15, %transport_fanout42_out6, %transport_fanout41_out15, %transport_fanout63_out13, %transport_fanout29_out21, %transport_fanout37_out15, %transport_fanout22_out13, %transport_fanout38_out9, %transport_fanout4_out38, %transport_fanout21_out35, %transport_fanout101_out5, %transport_fanout102_out5, %transport_fanout103_out5, %transport_fanout104_out5, %transport_fanout67_out20, %transport_fanout68_out18, %transport_fanout50_out9, %transport_fanout51_out5, %transport_fanout65_out4, %transport_fanout66_out4, %transport_fanout62_out3, %transport_fanout89_out6, %transport_fanout107_out3, %transport_fanout25_out8, %transport_fanout26_out6, %transport_fanout24_out5, %transport_fanout40_out8, %transport_fanout46_out25, %transport_fanout47_out17, %transport_fanout48_out8, %transport_fanout39_out11, %transport_fanout69_out6, %transport_fanout70_out8, %transport_fanout120_out2, %transport_fanout55_out20, %transport_fanout56_out19, %transport_fanout57_out16, %transport_fanout58_out15, %transport_fanout59_out6, %transport_fanout78_out6, %transport_fanout79_out6, %transport_fanout105_out4, %transport_fanout106_out4, %transport_fanout111_out3, %transport_fanout80_out7, %transport_fanout109_out4, %transport_fanout81_out3, %transport_fanout121_out2, %transport_fanout110_out3, %transport_fanout82_out4, %transport_fanout27_out12, %transport_fanout28_out12, %transport_fanout71_out10
        [{connectivity_table = ["1111111111111111111111111111111111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %typed_sync_i64_value = fabric.switch [spatial] %transport_fanout0_out17, %transport_fanout1_out14, %transport_fanout2_out14, %transport_fanout93_out18, %transport_fanout92_out18, %transport_fanout94_out13, %transport_fanout96_out11, %transport_fanout97_out11, %transport_fanout98_out7, %transport_fanout99_out5, %transport_fanout95_out10, %transport_fanout16_out1
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> !fabric.bits<64>
  %addr_add_lhs = fabric.switch [spatial] %transport_fanout41_out16, %transport_fanout11_out32, %transport_fanout30_out23, %transport_fanout31_out16, %transport_fanout49_out6, %transport_fanout50_out10, %transport_fanout29_out22, %transport_fanout32_out20, %transport_fanout51_out6, %transport_fanout4_out39, %transport_fanout21_out36
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %addr_add_rhs = fabric.switch [spatial] %transport_fanout33_out16, %transport_fanout32_out21, %transport_fanout11_out33, %transport_fanout30_out24, %transport_fanout41_out17, %transport_fanout62_out4, %transport_fanout63_out14, %transport_fanout64_out11, %transport_fanout37_out16, %transport_fanout50_out11, %transport_fanout51_out7, %transport_fanout49_out7, %transport_fanout29_out23
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %addr_mask_lhs = fabric.switch [spatial] %transport_fanout68_out19, %transport_fanout41_out18, %transport_fanout4_out40, %transport_fanout21_out37, %transport_fanout46_out26, %transport_fanout37_out17, %transport_fanout22_out14, %transport_fanout38_out10, %transport_fanout39_out12, %transport_fanout48_out9, %transport_fanout26_out7
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %addr_mask_rhs = fabric.switch [spatial] %transport_fanout32_out22, %transport_fanout33_out17, %transport_fanout30_out25, %transport_fanout31_out17, %transport_fanout25_out9, %transport_fanout70_out9, %transport_fanout46_out27, %transport_fanout34_out19, %transport_fanout35_out19, %transport_fanout36_out18, %transport_fanout48_out10, %transport_fanout26_out8
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %addr_unscale_lhs = fabric.switch [spatial] %transport_fanout11_out34, %transport_fanout40_out9, %transport_fanout22_out15, %transport_fanout4_out41, %transport_fanout49_out8, %transport_fanout50_out12, %transport_fanout51_out8
        [{connectivity_table = ["1111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %addr_unscale_rhs = fabric.switch [spatial] %transport_fanout30_out26, %transport_fanout40_out10, %transport_fanout43_out6, %transport_fanout32_out23, %transport_fanout33_out18, %transport_fanout34_out20, %transport_fanout35_out20, %transport_fanout36_out19
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %logic_shift_lhs = fabric.switch [spatial] %transport_fanout11_out35, %transport_fanout4_out42, %transport_fanout21_out38, %transport_fanout37_out18, %transport_fanout22_out16, %transport_fanout38_out11, %transport_fanout29_out24, %transport_fanout23_out4, %transport_fanout40_out11, %transport_fanout46_out28, %transport_fanout47_out18, %transport_fanout25_out10, %transport_fanout26_out9
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %logic_shift_rhs = fabric.switch [spatial] %transport_fanout30_out27, %transport_fanout40_out12, %transport_fanout32_out24, %transport_fanout43_out7, %transport_fanout44_out5, %transport_fanout45_out5, %transport_fanout34_out21, %transport_fanout35_out21, %transport_fanout36_out20, %transport_fanout52_out7, %transport_fanout53_out7, %transport_fanout54_out7, %transport_fanout60_out2, %transport_fanout61_out1
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %addr_shift_lhs = fabric.switch [spatial] %transport_fanout11_out36, %transport_fanout37_out19, %transport_fanout41_out19, %transport_fanout22_out17, %transport_fanout38_out12, %transport_fanout39_out13, %transport_fanout48_out11, %transport_fanout26_out10
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %addr_shift_rhs = fabric.switch [spatial] %transport_fanout30_out28, %transport_fanout32_out25, %transport_fanout34_out22, %transport_fanout35_out22, %transport_fanout36_out21, %transport_fanout43_out8, %transport_fanout44_out6, %transport_fanout45_out6, %transport_fanout33_out19, %transport_fanout48_out12, %transport_fanout26_out11, %transport_fanout52_out8, %transport_fanout53_out8, %transport_fanout54_out8, %transport_fanout60_out3, %transport_fanout61_out2
        [{connectivity_table = ["1111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %load0_addr = fabric.switch [spatial] %transport_fanout41_out20, %transport_fanout47_out19, %transport_fanout40_out13, %transport_fanout23_out5, %transport_fanout37_out20, %transport_fanout22_out18, %transport_fanout38_out13, %transport_fanout49_out9, %transport_fanout29_out25, %transport_fanout68_out20, %transport_fanout50_out13, %transport_fanout67_out21, %transport_fanout63_out15, %transport_fanout64_out12, %transport_fanout55_out21, %transport_fanout56_out20, %transport_fanout57_out17, %transport_fanout58_out16, %transport_fanout39_out14, %transport_fanout43_out9, %transport_fanout44_out7, %transport_fanout45_out7, %transport_fanout59_out7, %transport_fanout87_out2, %transport_fanout88_out2
        [{connectivity_table = ["1111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %load3_addr = fabric.switch [spatial] %transport_fanout42_out7, %transport_fanout37_out21, %transport_fanout41_out21, %transport_fanout49_out10, %transport_fanout29_out26, %transport_fanout68_out21, %transport_fanout67_out22, %transport_fanout63_out16, %transport_fanout64_out13, %transport_fanout59_out8, %transport_fanout87_out3, %transport_fanout88_out3
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %load4_addr = fabric.switch [spatial] %transport_fanout41_out22, %transport_fanout49_out11, %transport_fanout29_out27, %transport_fanout68_out22, %transport_fanout50_out14, %transport_fanout67_out23, %transport_fanout23_out6, %transport_fanout40_out14, %transport_fanout63_out17, %transport_fanout64_out14, %transport_fanout59_out9, %transport_fanout87_out4, %transport_fanout88_out4
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %load5_addr = fabric.switch [spatial] %transport_fanout41_out23, %transport_fanout49_out12, %transport_fanout29_out28, %transport_fanout68_out23, %transport_fanout50_out15, %transport_fanout67_out24, %transport_fanout23_out7, %transport_fanout40_out15, %transport_fanout63_out18, %transport_fanout64_out15, %transport_fanout59_out10, %transport_fanout87_out5, %transport_fanout88_out5
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %store0_addr = fabric.switch [spatial] %transport_fanout41_out24, %transport_fanout23_out8, %transport_fanout37_out22, %transport_fanout43_out10, %transport_fanout38_out14, %transport_fanout44_out8, %transport_fanout45_out8, %transport_fanout85_out2, %transport_fanout86_out2, %transport_fanout67_out25, %transport_fanout68_out24, %transport_fanout63_out19, %transport_fanout29_out29, %transport_fanout64_out16, %transport_fanout81_out4, %transport_fanout121_out3, %transport_fanout59_out11, %transport_fanout87_out6, %transport_fanout88_out6
        [{connectivity_table = ["1111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %store1_addr = fabric.switch [spatial] %transport_fanout31_out18, %transport_fanout41_out25, %transport_fanout23_out9, %transport_fanout37_out23, %transport_fanout43_out11, %transport_fanout44_out9, %transport_fanout45_out9, %transport_fanout85_out3, %transport_fanout86_out3, %transport_fanout67_out26, %transport_fanout68_out25, %transport_fanout63_out20, %transport_fanout29_out30, %transport_fanout64_out17, %transport_fanout59_out12, %transport_fanout87_out7, %transport_fanout88_out7
        [{connectivity_table = ["11111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_stream_lb = fabric.switch [spatial] %transport_fanout43_out12, %transport_fanout44_out10, %transport_fanout45_out10
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_stream_ub = fabric.switch [spatial] %transport_fanout50_out16, %transport_fanout51_out9, %transport_fanout49_out13
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_stream_step = fabric.switch [spatial] %transport_fanout43_out13, %transport_fanout44_out11, %transport_fanout45_out11
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_invariant_cond = fabric.switch [spatial] %transport_fanout73_out1, %transport_fanout5_out7
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_invariant0_value = fabric.switch [spatial] %transport_fanout11_out37, %transport_fanout30_out29, %transport_fanout31_out19, %transport_fanout42_out8, %transport_fanout32_out26, %transport_fanout34_out23, %transport_fanout35_out23, %transport_fanout36_out22, %transport_fanout33_out20, %transport_fanout43_out14, %transport_fanout44_out12, %transport_fanout45_out12
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %aux_invariant1_value = fabric.switch [spatial] %transport_fanout11_out38, %transport_fanout30_out30, %transport_fanout31_out20, %transport_fanout42_out9, %transport_fanout32_out27, %transport_fanout34_out24, %transport_fanout35_out24, %transport_fanout36_out23, %transport_fanout33_out21, %transport_fanout52_out9, %transport_fanout43_out15, %transport_fanout44_out13, %transport_fanout45_out13
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %store0_ctrl = fabric.switch [spatial] %transport_fanout8_out12, %transport_fanout20_out7, %transport_fanout112_out6, %transport_fanout114_out6, %transport_fanout116_out5, %transport_fanout18_out6, %transport_fanout117_out6, %transport_fanout83_out7, %transport_fanout84_out5
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %store1_ctrl = fabric.switch [spatial] %transport_fanout8_out13, %transport_fanout20_out8, %transport_fanout112_out7, %transport_fanout114_out7, %transport_fanout116_out6, %transport_fanout18_out7, %transport_fanout117_out7, %transport_fanout83_out8, %transport_fanout84_out6
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %data0, %done0, %data1, %done1, %data2, %done2, %data3, %done3, %data4, %done4, %data5, %done5, %store_done0, %store_done1 = fabric.mem [spatial] mgr(%mgr) load(%load0_addr, %transport_fanout8_out14, %load1_addr, %transport_fanout8_out15, %load2_addr, %transport_fanout8_out16, %load3_addr, %transport_fanout8_out17, %load4_addr, %transport_fanout8_out18, %load5_addr, %transport_fanout8_out19)
                                store(%store0_addr, %store0_value, %store0_ctrl, %store1_addr, %store1_value, %store1_ctrl)
        [{load_group_size = 6 : i32, store_group_size = 2 : i32, data_width = 32 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %mul_lhs_input = fabric.switch [spatial] %transport_fanout4_out43, %transport_fanout21_out39, %transport_fanout101_out6, %transport_fanout41_out26, %transport_fanout103_out6, %transport_fanout65_out5, %transport_fanout66_out5, %transport_fanout63_out21, %transport_fanout64_out18, %transport_fanout34_out25, %transport_fanout35_out25, %transport_fanout36_out24, %transport_fanout55_out22, %transport_fanout56_out21, %transport_fanout57_out18, %transport_fanout58_out17, %transport_fanout52_out10, %transport_fanout53_out9, %transport_fanout54_out9, %transport_fanout67_out27, %transport_fanout29_out31
        [{connectivity_table = ["111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mul_rhs_input = fabric.switch [spatial] %transport_fanout4_out44, %transport_fanout21_out40, %transport_fanout101_out7, %transport_fanout103_out7, %transport_fanout32_out28, %transport_fanout34_out26, %transport_fanout35_out26, %transport_fanout36_out25, %transport_fanout52_out11, %transport_fanout53_out10, %transport_fanout54_out10, %transport_fanout33_out22, %transport_fanout55_out23, %transport_fanout56_out22, %transport_fanout57_out19, %transport_fanout58_out18
        [{connectivity_table = ["1111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %reduction_input = fabric.switch [spatial] %transport_fanout4_out45, %transport_fanout108_out1, %transport_fanout49_out14
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %stream_sum_lhs = fabric.switch [spatial] %reduction_input, %transport_fanout37_out24, %transport_fanout22_out19, %transport_fanout38_out15, %transport_fanout50_out17, %transport_fanout51_out10, %transport_fanout34_out27, %transport_fanout35_out27, %transport_fanout36_out26, %transport_fanout55_out24, %transport_fanout56_out23, %transport_fanout57_out20, %transport_fanout58_out19, %transport_fanout59_out13
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %stream_sum_rhs = fabric.switch [spatial] %transport_fanout37_out25, %transport_fanout33_out23, %transport_fanout32_out29, %transport_fanout36_out27, %transport_fanout62_out5, %transport_fanout63_out22, %transport_fanout52_out12, %transport_fanout53_out11, %transport_fanout54_out11, %transport_fanout34_out28, %transport_fanout35_out28, %transport_fanout55_out25, %transport_fanout56_out24, %transport_fanout57_out21, %transport_fanout58_out20, %transport_fanout40_out16, %transport_fanout59_out14
        [{connectivity_table = ["11111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %scan_init = fabric.switch [spatial] %transport_fanout11_out39, %transport_fanout43_out16, %transport_fanout44_out14, %transport_fanout45_out14
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %scan_scale = fabric.switch [spatial] %transport_fanout30_out31, %transport_fanout43_out17, %transport_fanout44_out15, %transport_fanout45_out15
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_lhs = fabric.switch [spatial] %transport_fanout37_out26, %transport_fanout4_out46, %transport_fanout101_out8, %transport_fanout103_out8, %transport_fanout32_out30, %transport_fanout75_out4
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_rhs = fabric.switch [spatial] %transport_fanout4_out47, %transport_fanout21_out41, %transport_fanout102_out6, %transport_fanout104_out6, %transport_fanout32_out31
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_lhs_aux = fabric.switch [spatial] %transport_fanout22_out20, %transport_fanout38_out16, %transport_fanout37_out27, %transport_fanout4_out48, %transport_fanout21_out42, %transport_fanout101_out9, %transport_fanout103_out9, %transport_fanout32_out32, %transport_fanout75_out5
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_rhs_aux = fabric.switch [spatial] %transport_fanout21_out43, %transport_fanout4_out49, %transport_fanout102_out7, %transport_fanout104_out7, %transport_fanout32_out33, %transport_fanout33_out24, %transport_fanout34_out29
        [{connectivity_table = ["1111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_diff_lhs = fabric.switch [spatial] %transport_fanout11_out40, %transport_fanout4_out50
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_diff_rhs = fabric.switch [spatial] %transport_fanout30_out32, %transport_fanout33_out25, %transport_fanout21_out44, %transport_fanout122_out0
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_diff_aux_lhs = fabric.switch [spatial] %transport_fanout21_out45, %transport_fanout4_out51, %transport_fanout11_out41
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_diff_aux_rhs = fabric.switch [spatial] %transport_fanout34_out30, %transport_fanout33_out26, %transport_fanout35_out29, %transport_fanout36_out28, %transport_fanout52_out13, %transport_fanout53_out12, %transport_fanout54_out12, %transport_fanout30_out33, %transport_fanout21_out46, %transport_fanout122_out1
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_div_lhs = fabric.switch [spatial] %transport_fanout21_out47, %transport_fanout4_out52
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_div_rhs = fabric.switch [spatial] %transport_fanout101_out10, %transport_fanout33_out27, %transport_fanout32_out34
        [{connectivity_table = ["111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %fp_invariant_value = fabric.switch [spatial] %transport_fanout30_out34, %transport_fanout43_out18, %transport_fanout44_out16, %transport_fanout45_out16
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %bit_invariant_value = fabric.switch [spatial] %transport_fanout42_out10, %transport_fanout32_out35
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %bit_invariant_aux0_value = fabric.switch [spatial] %transport_fanout31_out21, %transport_fanout33_out28
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %bit_invariant_aux1_value = fabric.switch [spatial] %transport_fanout30_out35, %transport_fanout32_out36, %transport_fanout43_out19, %transport_fanout44_out17, %transport_fanout45_out17
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %scaled_reduction_lhs = fabric.switch [spatial] %transport_fanout37_out28, %transport_fanout78_out7, %transport_fanout79_out7, %transport_fanout21_out48, %transport_fanout102_out8, %transport_fanout104_out8, %transport_fanout4_out53, %transport_fanout34_out31, %transport_fanout35_out30, %transport_fanout36_out29, %transport_fanout52_out14, %transport_fanout53_out13, %transport_fanout54_out13, %transport_fanout111_out4, %transport_fanout32_out37
        [{connectivity_table = ["111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %scaled_reduction_rhs = fabric.switch [spatial] %transport_fanout32_out38, %transport_fanout103_out10, %transport_fanout104_out9, %transport_fanout21_out49, %transport_fanout102_out9, %transport_fanout38_out17, %transport_fanout22_out21, %transport_fanout52_out15, %transport_fanout53_out14, %transport_fanout54_out14, %transport_fanout111_out5, %transport_fanout4_out54
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %scaled_reduction_aux_lhs = fabric.switch [spatial] %transport_fanout37_out29, %transport_fanout78_out8, %transport_fanout79_out8, %transport_fanout21_out50, %transport_fanout102_out10, %transport_fanout104_out10, %transport_fanout4_out55, %transport_fanout34_out32, %transport_fanout35_out31, %transport_fanout36_out30, %transport_fanout52_out16, %transport_fanout53_out15, %transport_fanout54_out15, %transport_fanout111_out6, %transport_fanout32_out39
        [{connectivity_table = ["111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %scaled_reduction_aux_rhs = fabric.switch [spatial] %transport_fanout32_out40, %transport_fanout103_out11, %transport_fanout104_out11, %transport_fanout21_out51, %transport_fanout102_out11, %transport_fanout38_out18, %transport_fanout22_out22, %transport_fanout52_out17, %transport_fanout53_out16, %transport_fanout54_out16, %transport_fanout111_out7, %transport_fanout4_out56
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac_lhs = fabric.switch [spatial] %transport_fanout11_out42, %transport_fanout4_out57, %transport_fanout101_out11, %transport_fanout103_out12, %transport_fanout105_out5, %transport_fanout106_out5, %transport_fanout80_out8, %transport_fanout33_out29, %transport_fanout34_out33, %transport_fanout35_out32, %transport_fanout21_out52, %transport_fanout36_out31, %transport_fanout32_out41
        [{connectivity_table = ["1111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac_rhs = fabric.switch [spatial] %transport_fanout30_out36, %transport_fanout21_out53, %transport_fanout101_out12, %transport_fanout102_out12, %transport_fanout104_out12, %transport_fanout105_out6, %transport_fanout106_out6, %transport_fanout4_out58, %transport_fanout22_out23, %transport_fanout38_out19
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac_acc = fabric.switch [spatial] %transport_fanout31_out22, %transport_fanout37_out30, %transport_fanout22_out24, %transport_fanout80_out9, %transport_fanout38_out20, %transport_fanout4_out59
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac1_lhs = fabric.switch [spatial] %transport_fanout11_out43, %transport_fanout101_out13, %transport_fanout103_out13, %transport_fanout4_out60, %transport_fanout105_out7, %transport_fanout106_out7, %transport_fanout33_out30, %transport_fanout34_out34, %transport_fanout35_out33, %transport_fanout36_out32, %transport_fanout32_out42
        [{connectivity_table = ["11111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac1_rhs = fabric.switch [spatial] %transport_fanout30_out37, %transport_fanout102_out13, %transport_fanout104_out13, %transport_fanout21_out54, %transport_fanout105_out8, %transport_fanout106_out8, %transport_fanout22_out25, %transport_fanout38_out21, %transport_fanout37_out31
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac1_acc = fabric.switch [spatial] %transport_fanout31_out23, %transport_fanout74_out4, %transport_fanout80_out10, %transport_fanout37_out32, %transport_fanout22_out26, %transport_fanout38_out22
        [{connectivity_table = ["111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac2_lhs = fabric.switch [spatial] %transport_fanout11_out44, %transport_fanout4_out61, %transport_fanout101_out14, %transport_fanout103_out14, %transport_fanout33_out31, %transport_fanout34_out35, %transport_fanout35_out34, %transport_fanout36_out33, %transport_fanout32_out43
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac2_rhs = fabric.switch [spatial] %transport_fanout30_out38, %transport_fanout4_out62, %transport_fanout21_out55, %transport_fanout102_out14, %transport_fanout104_out14, %transport_fanout22_out27, %transport_fanout38_out23, %transport_fanout37_out33
        [{connectivity_table = ["11111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac2_acc = fabric.switch [spatial] %transport_fanout75_out6, %transport_fanout74_out5, %transport_fanout80_out11, %transport_fanout22_out28, %transport_fanout38_out24
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac3_lhs = fabric.switch [spatial] %transport_fanout11_out45, %transport_fanout4_out63, %transport_fanout101_out15, %transport_fanout103_out15, %transport_fanout33_out32, %transport_fanout34_out36, %transport_fanout35_out35, %transport_fanout36_out34, %transport_fanout32_out44
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac3_rhs = fabric.switch [spatial] %transport_fanout30_out39, %transport_fanout4_out64, %transport_fanout21_out56, %transport_fanout102_out15, %transport_fanout104_out15, %transport_fanout22_out29, %transport_fanout38_out25, %transport_fanout37_out34, %transport_fanout78_out9, %transport_fanout79_out9
        [{connectivity_table = ["1111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %mac3_acc = fabric.switch [spatial] %transport_fanout76_out4, %transport_fanout75_out7, %transport_fanout74_out6, %transport_fanout80_out12, %transport_fanout22_out30, %transport_fanout38_out26, %transport_fanout103_out16
        [{connectivity_table = ["1111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %bit_carry_cond = fabric.switch [spatial] %transport_fanout11_out46, %transport_fanout5_out8
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %bit_carry_init = fabric.switch [spatial] %transport_fanout30_out40, %transport_fanout31_out24, %transport_fanout43_out20, %transport_fanout45_out18
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %bit_carry_next = fabric.switch [spatial] %transport_fanout31_out25, %transport_fanout23_out10, %transport_fanout74_out7, %transport_fanout75_out8, %transport_fanout67_out28, %transport_fanout39_out15, %transport_fanout29_out32, %transport_fanout76_out5, %transport_fanout77_out4, %transport_fanout4_out65, %transport_fanout38_out27, %transport_fanout48_out13, %transport_fanout26_out12, %transport_fanout79_out10
        [{connectivity_table = ["11111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %state_carry_cond = fabric.switch [spatial] %transport_fanout5_out9, %transport_fanout11_out47
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %state_carry_init = fabric.switch [spatial] %transport_fanout11_out48, %transport_fanout30_out41, %transport_fanout31_out26, %transport_fanout42_out11, %transport_fanout43_out21, %transport_fanout44_out18, %transport_fanout45_out19, %transport_fanout4_out66, %transport_fanout21_out57
        [{connectivity_table = ["111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %state_carry_next = fabric.switch [spatial] %transport_fanout74_out8, %transport_fanout75_out9, %transport_fanout76_out6, %transport_fanout77_out5, %transport_fanout22_out31, %transport_fanout37_out35, %transport_fanout67_out29, %transport_fanout4_out67, %transport_fanout29_out33, %transport_fanout48_out14, %transport_fanout26_out13, %transport_fanout79_out11
        [{connectivity_table = ["111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>
  %scan_feedback, %scan_store_value = fabric.switch [spatial] %transport_fanout29_out34, %transport_fanout78_out10, %transport_fanout74_out9, %transport_fanout75_out10, %transport_fanout76_out7, %transport_fanout77_out6, %transport_fanout22_out32, %transport_fanout38_out28, %transport_fanout107_out4, %transport_fanout39_out16, %transport_fanout67_out30, %transport_fanout68_out26, %transport_fanout50_out18, %transport_fanout51_out11, %control_muxed, %transport_fanout25_out11, %transport_fanout48_out15, %transport_fanout26_out14, %transport_fanout79_out12, %transport_fanout89_out7
        [{connectivity_table = ["11111111111111111111", "00111100000000000000"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  %sync_aux_done = fabric.switch [spatial] %transport_fanout19_out13, %transport_fanout112_out8, %transport_fanout114_out8, %transport_fanout116_out7, %transport_fanout18_out8, %transport_fanout117_out8, %transport_fanout113_out6
        [{connectivity_table = ["1111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> !fabric.bits<0>
  %transport_fanout0_out0, %transport_fanout0_out1, %transport_fanout0_out2, %transport_fanout0_out3, %transport_fanout0_out4, %transport_fanout0_out5, %transport_fanout0_out6, %transport_fanout0_out7, %transport_fanout0_out8, %transport_fanout0_out9, %transport_fanout0_out10, %transport_fanout0_out11, %transport_fanout0_out12, %transport_fanout0_out13, %transport_fanout0_out14, %transport_fanout0_out15, %transport_fanout0_out16, %transport_fanout0_out17 = fabric.switch [spatial] %i64a
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout1_out0, %transport_fanout1_out1, %transport_fanout1_out2, %transport_fanout1_out3, %transport_fanout1_out4, %transport_fanout1_out5, %transport_fanout1_out6, %transport_fanout1_out7, %transport_fanout1_out8, %transport_fanout1_out9, %transport_fanout1_out10, %transport_fanout1_out11, %transport_fanout1_out12, %transport_fanout1_out13, %transport_fanout1_out14 = fabric.switch [spatial] %i64b
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout2_out0, %transport_fanout2_out1, %transport_fanout2_out2, %transport_fanout2_out3, %transport_fanout2_out4, %transport_fanout2_out5, %transport_fanout2_out6, %transport_fanout2_out7, %transport_fanout2_out8, %transport_fanout2_out9, %transport_fanout2_out10, %transport_fanout2_out11, %transport_fanout2_out12, %transport_fanout2_out13, %transport_fanout2_out14 = fabric.switch [spatial] %i64c
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout3_out0, %transport_fanout3_out1 = fabric.switch [spatial] %gate_cond
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout4_out0, %transport_fanout4_out1, %transport_fanout4_out2, %transport_fanout4_out3, %transport_fanout4_out4, %transport_fanout4_out5, %transport_fanout4_out6, %transport_fanout4_out7, %transport_fanout4_out8, %transport_fanout4_out9, %transport_fanout4_out10, %transport_fanout4_out11, %transport_fanout4_out12, %transport_fanout4_out13, %transport_fanout4_out14, %transport_fanout4_out15, %transport_fanout4_out16, %transport_fanout4_out17, %transport_fanout4_out18, %transport_fanout4_out19, %transport_fanout4_out20, %transport_fanout4_out21, %transport_fanout4_out22, %transport_fanout4_out23, %transport_fanout4_out24, %transport_fanout4_out25, %transport_fanout4_out26, %transport_fanout4_out27, %transport_fanout4_out28, %transport_fanout4_out29, %transport_fanout4_out30, %transport_fanout4_out31, %transport_fanout4_out32, %transport_fanout4_out33, %transport_fanout4_out34, %transport_fanout4_out35, %transport_fanout4_out36, %transport_fanout4_out37, %transport_fanout4_out38, %transport_fanout4_out39, %transport_fanout4_out40, %transport_fanout4_out41, %transport_fanout4_out42, %transport_fanout4_out43, %transport_fanout4_out44, %transport_fanout4_out45, %transport_fanout4_out46, %transport_fanout4_out47, %transport_fanout4_out48, %transport_fanout4_out49, %transport_fanout4_out50, %transport_fanout4_out51, %transport_fanout4_out52, %transport_fanout4_out53, %transport_fanout4_out54, %transport_fanout4_out55, %transport_fanout4_out56, %transport_fanout4_out57, %transport_fanout4_out58, %transport_fanout4_out59, %transport_fanout4_out60, %transport_fanout4_out61, %transport_fanout4_out62, %transport_fanout4_out63, %transport_fanout4_out64, %transport_fanout4_out65, %transport_fanout4_out66, %transport_fanout4_out67 = fabric.switch [spatial] %data0
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout5_out0, %transport_fanout5_out1, %transport_fanout5_out2, %transport_fanout5_out3, %transport_fanout5_out4, %transport_fanout5_out5, %transport_fanout5_out6, %transport_fanout5_out7, %transport_fanout5_out8, %transport_fanout5_out9 = fabric.switch [spatial] %fp_gate
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout6_out0, %transport_fanout6_out1, %transport_fanout6_out2, %transport_fanout6_out3, %transport_fanout6_out4 = fabric.switch [spatial] %aux_invariant_cond
         [{connectivity_table = ["1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout7_out0, %transport_fanout7_out1 = fabric.switch [spatial] %bit_invariant_aux1_value
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout8_out0, %transport_fanout8_out1, %transport_fanout8_out2, %transport_fanout8_out3, %transport_fanout8_out4, %transport_fanout8_out5, %transport_fanout8_out6, %transport_fanout8_out7, %transport_fanout8_out8, %transport_fanout8_out9, %transport_fanout8_out10, %transport_fanout8_out11, %transport_fanout8_out12, %transport_fanout8_out13, %transport_fanout8_out14, %transport_fanout8_out15, %transport_fanout8_out16, %transport_fanout8_out17, %transport_fanout8_out18, %transport_fanout8_out19 = fabric.switch [spatial] %ctrl
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout9_out0, %transport_fanout9_out1, %transport_fanout9_out2, %transport_fanout9_out3 = fabric.switch [spatial] %minmax_lhs
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout10_out0, %transport_fanout10_out1, %transport_fanout10_out2, %transport_fanout10_out3 = fabric.switch [spatial] %minmax_rhs
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout11_out0, %transport_fanout11_out1, %transport_fanout11_out2, %transport_fanout11_out3, %transport_fanout11_out4, %transport_fanout11_out5, %transport_fanout11_out6, %transport_fanout11_out7, %transport_fanout11_out8, %transport_fanout11_out9, %transport_fanout11_out10, %transport_fanout11_out11, %transport_fanout11_out12, %transport_fanout11_out13, %transport_fanout11_out14, %transport_fanout11_out15, %transport_fanout11_out16, %transport_fanout11_out17, %transport_fanout11_out18, %transport_fanout11_out19, %transport_fanout11_out20, %transport_fanout11_out21, %transport_fanout11_out22, %transport_fanout11_out23, %transport_fanout11_out24, %transport_fanout11_out25, %transport_fanout11_out26, %transport_fanout11_out27, %transport_fanout11_out28, %transport_fanout11_out29, %transport_fanout11_out30, %transport_fanout11_out31, %transport_fanout11_out32, %transport_fanout11_out33, %transport_fanout11_out34, %transport_fanout11_out35, %transport_fanout11_out36, %transport_fanout11_out37, %transport_fanout11_out38, %transport_fanout11_out39, %transport_fanout11_out40, %transport_fanout11_out41, %transport_fanout11_out42, %transport_fanout11_out43, %transport_fanout11_out44, %transport_fanout11_out45, %transport_fanout11_out46, %transport_fanout11_out47, %transport_fanout11_out48 = fabric.switch [spatial] %i32a
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout12_out0, %transport_fanout12_out1, %transport_fanout12_out2 = fabric.switch [spatial] %cmp_lhs
         [{connectivity_table = ["1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout13_out0, %transport_fanout13_out1, %transport_fanout13_out2 = fabric.switch [spatial] %cmp_rhs
         [{connectivity_table = ["1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout14_out0, %transport_fanout14_out1 = fabric.switch [spatial] %cmp64_lhs
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout15_out0, %transport_fanout15_out1 = fabric.switch [spatial] %cmp64_rhs
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout16_out0, %transport_fanout16_out1 = fabric.switch [spatial] %cmpi64_pred
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout17_out0, %transport_fanout17_out1 = fabric.switch [spatial] %demux_sel
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout18_out0, %transport_fanout18_out1, %transport_fanout18_out2, %transport_fanout18_out3, %transport_fanout18_out4, %transport_fanout18_out5, %transport_fanout18_out6, %transport_fanout18_out7, %transport_fanout18_out8 = fabric.switch [spatial] %done4
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout19_out0, %transport_fanout19_out1, %transport_fanout19_out2, %transport_fanout19_out3, %transport_fanout19_out4, %transport_fanout19_out5, %transport_fanout19_out6, %transport_fanout19_out7, %transport_fanout19_out8, %transport_fanout19_out9, %transport_fanout19_out10, %transport_fanout19_out11, %transport_fanout19_out12, %transport_fanout19_out13 = fabric.switch [spatial] %store_done0
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout20_out0, %transport_fanout20_out1, %transport_fanout20_out2, %transport_fanout20_out3, %transport_fanout20_out4, %transport_fanout20_out5, %transport_fanout20_out6, %transport_fanout20_out7, %transport_fanout20_out8 = fabric.switch [spatial] %done0
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout21_out0, %transport_fanout21_out1, %transport_fanout21_out2, %transport_fanout21_out3, %transport_fanout21_out4, %transport_fanout21_out5, %transport_fanout21_out6, %transport_fanout21_out7, %transport_fanout21_out8, %transport_fanout21_out9, %transport_fanout21_out10, %transport_fanout21_out11, %transport_fanout21_out12, %transport_fanout21_out13, %transport_fanout21_out14, %transport_fanout21_out15, %transport_fanout21_out16, %transport_fanout21_out17, %transport_fanout21_out18, %transport_fanout21_out19, %transport_fanout21_out20, %transport_fanout21_out21, %transport_fanout21_out22, %transport_fanout21_out23, %transport_fanout21_out24, %transport_fanout21_out25, %transport_fanout21_out26, %transport_fanout21_out27, %transport_fanout21_out28, %transport_fanout21_out29, %transport_fanout21_out30, %transport_fanout21_out31, %transport_fanout21_out32, %transport_fanout21_out33, %transport_fanout21_out34, %transport_fanout21_out35, %transport_fanout21_out36, %transport_fanout21_out37, %transport_fanout21_out38, %transport_fanout21_out39, %transport_fanout21_out40, %transport_fanout21_out41, %transport_fanout21_out42, %transport_fanout21_out43, %transport_fanout21_out44, %transport_fanout21_out45, %transport_fanout21_out46, %transport_fanout21_out47, %transport_fanout21_out48, %transport_fanout21_out49, %transport_fanout21_out50, %transport_fanout21_out51, %transport_fanout21_out52, %transport_fanout21_out53, %transport_fanout21_out54, %transport_fanout21_out55, %transport_fanout21_out56, %transport_fanout21_out57 = fabric.switch [spatial] %data1
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout22_out0, %transport_fanout22_out1, %transport_fanout22_out2, %transport_fanout22_out3, %transport_fanout22_out4, %transport_fanout22_out5, %transport_fanout22_out6, %transport_fanout22_out7, %transport_fanout22_out8, %transport_fanout22_out9, %transport_fanout22_out10, %transport_fanout22_out11, %transport_fanout22_out12, %transport_fanout22_out13, %transport_fanout22_out14, %transport_fanout22_out15, %transport_fanout22_out16, %transport_fanout22_out17, %transport_fanout22_out18, %transport_fanout22_out19, %transport_fanout22_out20, %transport_fanout22_out21, %transport_fanout22_out22, %transport_fanout22_out23, %transport_fanout22_out24, %transport_fanout22_out25, %transport_fanout22_out26, %transport_fanout22_out27, %transport_fanout22_out28, %transport_fanout22_out29, %transport_fanout22_out30, %transport_fanout22_out31, %transport_fanout22_out32 = fabric.switch [spatial] %bit_carry
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout23_out0, %transport_fanout23_out1, %transport_fanout23_out2, %transport_fanout23_out3, %transport_fanout23_out4, %transport_fanout23_out5, %transport_fanout23_out6, %transport_fanout23_out7, %transport_fanout23_out8, %transport_fanout23_out9, %transport_fanout23_out10 = fabric.switch [spatial] %addr_unscaled
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout24_out0, %transport_fanout24_out1, %transport_fanout24_out2, %transport_fanout24_out3, %transport_fanout24_out4, %transport_fanout24_out5 = fabric.switch [spatial] %logic_shifted
         [{connectivity_table = ["1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout25_out0, %transport_fanout25_out1, %transport_fanout25_out2, %transport_fanout25_out3, %transport_fanout25_out4, %transport_fanout25_out5, %transport_fanout25_out6, %transport_fanout25_out7, %transport_fanout25_out8, %transport_fanout25_out9, %transport_fanout25_out10, %transport_fanout25_out11 = fabric.switch [spatial] %int_xor
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout26_out0, %transport_fanout26_out1, %transport_fanout26_out2, %transport_fanout26_out3, %transport_fanout26_out4, %transport_fanout26_out5, %transport_fanout26_out6, %transport_fanout26_out7, %transport_fanout26_out8, %transport_fanout26_out9, %transport_fanout26_out10, %transport_fanout26_out11, %transport_fanout26_out12, %transport_fanout26_out13, %transport_fanout26_out14 = fabric.switch [spatial] %aux_xor
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout27_out0, %transport_fanout27_out1, %transport_fanout27_out2, %transport_fanout27_out3, %transport_fanout27_out4, %transport_fanout27_out5, %transport_fanout27_out6, %transport_fanout27_out7, %transport_fanout27_out8, %transport_fanout27_out9, %transport_fanout27_out10, %transport_fanout27_out11, %transport_fanout27_out12 = fabric.switch [spatial] %cmpi_pred
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout28_out0, %transport_fanout28_out1, %transport_fanout28_out2, %transport_fanout28_out3, %transport_fanout28_out4, %transport_fanout28_out5, %transport_fanout28_out6, %transport_fanout28_out7, %transport_fanout28_out8, %transport_fanout28_out9, %transport_fanout28_out10, %transport_fanout28_out11, %transport_fanout28_out12 = fabric.switch [spatial] %cmpi_pred_aux
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout29_out0, %transport_fanout29_out1, %transport_fanout29_out2, %transport_fanout29_out3, %transport_fanout29_out4, %transport_fanout29_out5, %transport_fanout29_out6, %transport_fanout29_out7, %transport_fanout29_out8, %transport_fanout29_out9, %transport_fanout29_out10, %transport_fanout29_out11, %transport_fanout29_out12, %transport_fanout29_out13, %transport_fanout29_out14, %transport_fanout29_out15, %transport_fanout29_out16, %transport_fanout29_out17, %transport_fanout29_out18, %transport_fanout29_out19, %transport_fanout29_out20, %transport_fanout29_out21, %transport_fanout29_out22, %transport_fanout29_out23, %transport_fanout29_out24, %transport_fanout29_out25, %transport_fanout29_out26, %transport_fanout29_out27, %transport_fanout29_out28, %transport_fanout29_out29, %transport_fanout29_out30, %transport_fanout29_out31, %transport_fanout29_out32, %transport_fanout29_out33, %transport_fanout29_out34 = fabric.switch [spatial] %running
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout30_out0, %transport_fanout30_out1, %transport_fanout30_out2, %transport_fanout30_out3, %transport_fanout30_out4, %transport_fanout30_out5, %transport_fanout30_out6, %transport_fanout30_out7, %transport_fanout30_out8, %transport_fanout30_out9, %transport_fanout30_out10, %transport_fanout30_out11, %transport_fanout30_out12, %transport_fanout30_out13, %transport_fanout30_out14, %transport_fanout30_out15, %transport_fanout30_out16, %transport_fanout30_out17, %transport_fanout30_out18, %transport_fanout30_out19, %transport_fanout30_out20, %transport_fanout30_out21, %transport_fanout30_out22, %transport_fanout30_out23, %transport_fanout30_out24, %transport_fanout30_out25, %transport_fanout30_out26, %transport_fanout30_out27, %transport_fanout30_out28, %transport_fanout30_out29, %transport_fanout30_out30, %transport_fanout30_out31, %transport_fanout30_out32, %transport_fanout30_out33, %transport_fanout30_out34, %transport_fanout30_out35, %transport_fanout30_out36, %transport_fanout30_out37, %transport_fanout30_out38, %transport_fanout30_out39, %transport_fanout30_out40, %transport_fanout30_out41 = fabric.switch [spatial] %i32b
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout31_out0, %transport_fanout31_out1, %transport_fanout31_out2, %transport_fanout31_out3, %transport_fanout31_out4, %transport_fanout31_out5, %transport_fanout31_out6, %transport_fanout31_out7, %transport_fanout31_out8, %transport_fanout31_out9, %transport_fanout31_out10, %transport_fanout31_out11, %transport_fanout31_out12, %transport_fanout31_out13, %transport_fanout31_out14, %transport_fanout31_out15, %transport_fanout31_out16, %transport_fanout31_out17, %transport_fanout31_out18, %transport_fanout31_out19, %transport_fanout31_out20, %transport_fanout31_out21, %transport_fanout31_out22, %transport_fanout31_out23, %transport_fanout31_out24, %transport_fanout31_out25, %transport_fanout31_out26 = fabric.switch [spatial] %i32c
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout32_out0, %transport_fanout32_out1, %transport_fanout32_out2, %transport_fanout32_out3, %transport_fanout32_out4, %transport_fanout32_out5, %transport_fanout32_out6, %transport_fanout32_out7, %transport_fanout32_out8, %transport_fanout32_out9, %transport_fanout32_out10, %transport_fanout32_out11, %transport_fanout32_out12, %transport_fanout32_out13, %transport_fanout32_out14, %transport_fanout32_out15, %transport_fanout32_out16, %transport_fanout32_out17, %transport_fanout32_out18, %transport_fanout32_out19, %transport_fanout32_out20, %transport_fanout32_out21, %transport_fanout32_out22, %transport_fanout32_out23, %transport_fanout32_out24, %transport_fanout32_out25, %transport_fanout32_out26, %transport_fanout32_out27, %transport_fanout32_out28, %transport_fanout32_out29, %transport_fanout32_out30, %transport_fanout32_out31, %transport_fanout32_out32, %transport_fanout32_out33, %transport_fanout32_out34, %transport_fanout32_out35, %transport_fanout32_out36, %transport_fanout32_out37, %transport_fanout32_out38, %transport_fanout32_out39, %transport_fanout32_out40, %transport_fanout32_out41, %transport_fanout32_out42, %transport_fanout32_out43, %transport_fanout32_out44 = fabric.switch [spatial] %reduction_scale
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout33_out0, %transport_fanout33_out1, %transport_fanout33_out2, %transport_fanout33_out3, %transport_fanout33_out4, %transport_fanout33_out5, %transport_fanout33_out6, %transport_fanout33_out7, %transport_fanout33_out8, %transport_fanout33_out9, %transport_fanout33_out10, %transport_fanout33_out11, %transport_fanout33_out12, %transport_fanout33_out13, %transport_fanout33_out14, %transport_fanout33_out15, %transport_fanout33_out16, %transport_fanout33_out17, %transport_fanout33_out18, %transport_fanout33_out19, %transport_fanout33_out20, %transport_fanout33_out21, %transport_fanout33_out22, %transport_fanout33_out23, %transport_fanout33_out24, %transport_fanout33_out25, %transport_fanout33_out26, %transport_fanout33_out27, %transport_fanout33_out28, %transport_fanout33_out29, %transport_fanout33_out30, %transport_fanout33_out31, %transport_fanout33_out32 = fabric.switch [spatial] %fp_invariant
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout34_out0, %transport_fanout34_out1, %transport_fanout34_out2, %transport_fanout34_out3, %transport_fanout34_out4, %transport_fanout34_out5, %transport_fanout34_out6, %transport_fanout34_out7, %transport_fanout34_out8, %transport_fanout34_out9, %transport_fanout34_out10, %transport_fanout34_out11, %transport_fanout34_out12, %transport_fanout34_out13, %transport_fanout34_out14, %transport_fanout34_out15, %transport_fanout34_out16, %transport_fanout34_out17, %transport_fanout34_out18, %transport_fanout34_out19, %transport_fanout34_out20, %transport_fanout34_out21, %transport_fanout34_out22, %transport_fanout34_out23, %transport_fanout34_out24, %transport_fanout34_out25, %transport_fanout34_out26, %transport_fanout34_out27, %transport_fanout34_out28, %transport_fanout34_out29, %transport_fanout34_out30, %transport_fanout34_out31, %transport_fanout34_out32, %transport_fanout34_out33, %transport_fanout34_out34, %transport_fanout34_out35, %transport_fanout34_out36 = fabric.switch [spatial] %bit_invariant
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout35_out0, %transport_fanout35_out1, %transport_fanout35_out2, %transport_fanout35_out3, %transport_fanout35_out4, %transport_fanout35_out5, %transport_fanout35_out6, %transport_fanout35_out7, %transport_fanout35_out8, %transport_fanout35_out9, %transport_fanout35_out10, %transport_fanout35_out11, %transport_fanout35_out12, %transport_fanout35_out13, %transport_fanout35_out14, %transport_fanout35_out15, %transport_fanout35_out16, %transport_fanout35_out17, %transport_fanout35_out18, %transport_fanout35_out19, %transport_fanout35_out20, %transport_fanout35_out21, %transport_fanout35_out22, %transport_fanout35_out23, %transport_fanout35_out24, %transport_fanout35_out25, %transport_fanout35_out26, %transport_fanout35_out27, %transport_fanout35_out28, %transport_fanout35_out29, %transport_fanout35_out30, %transport_fanout35_out31, %transport_fanout35_out32, %transport_fanout35_out33, %transport_fanout35_out34, %transport_fanout35_out35 = fabric.switch [spatial] %bit_invariant_aux0
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout36_out0, %transport_fanout36_out1, %transport_fanout36_out2, %transport_fanout36_out3, %transport_fanout36_out4, %transport_fanout36_out5, %transport_fanout36_out6, %transport_fanout36_out7, %transport_fanout36_out8, %transport_fanout36_out9, %transport_fanout36_out10, %transport_fanout36_out11, %transport_fanout36_out12, %transport_fanout36_out13, %transport_fanout36_out14, %transport_fanout36_out15, %transport_fanout36_out16, %transport_fanout36_out17, %transport_fanout36_out18, %transport_fanout36_out19, %transport_fanout36_out20, %transport_fanout36_out21, %transport_fanout36_out22, %transport_fanout36_out23, %transport_fanout36_out24, %transport_fanout36_out25, %transport_fanout36_out26, %transport_fanout36_out27, %transport_fanout36_out28, %transport_fanout36_out29, %transport_fanout36_out30, %transport_fanout36_out31, %transport_fanout36_out32, %transport_fanout36_out33, %transport_fanout36_out34 = fabric.switch [spatial] %bit_invariant_aux1
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout37_out0, %transport_fanout37_out1, %transport_fanout37_out2, %transport_fanout37_out3, %transport_fanout37_out4, %transport_fanout37_out5, %transport_fanout37_out6, %transport_fanout37_out7, %transport_fanout37_out8, %transport_fanout37_out9, %transport_fanout37_out10, %transport_fanout37_out11, %transport_fanout37_out12, %transport_fanout37_out13, %transport_fanout37_out14, %transport_fanout37_out15, %transport_fanout37_out16, %transport_fanout37_out17, %transport_fanout37_out18, %transport_fanout37_out19, %transport_fanout37_out20, %transport_fanout37_out21, %transport_fanout37_out22, %transport_fanout37_out23, %transport_fanout37_out24, %transport_fanout37_out25, %transport_fanout37_out26, %transport_fanout37_out27, %transport_fanout37_out28, %transport_fanout37_out29, %transport_fanout37_out30, %transport_fanout37_out31, %transport_fanout37_out32, %transport_fanout37_out33, %transport_fanout37_out34, %transport_fanout37_out35 = fabric.switch [spatial] %carried_scan
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout38_out0, %transport_fanout38_out1, %transport_fanout38_out2, %transport_fanout38_out3, %transport_fanout38_out4, %transport_fanout38_out5, %transport_fanout38_out6, %transport_fanout38_out7, %transport_fanout38_out8, %transport_fanout38_out9, %transport_fanout38_out10, %transport_fanout38_out11, %transport_fanout38_out12, %transport_fanout38_out13, %transport_fanout38_out14, %transport_fanout38_out15, %transport_fanout38_out16, %transport_fanout38_out17, %transport_fanout38_out18, %transport_fanout38_out19, %transport_fanout38_out20, %transport_fanout38_out21, %transport_fanout38_out22, %transport_fanout38_out23, %transport_fanout38_out24, %transport_fanout38_out25, %transport_fanout38_out26, %transport_fanout38_out27, %transport_fanout38_out28 = fabric.switch [spatial] %state_carry
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout39_out0, %transport_fanout39_out1, %transport_fanout39_out2, %transport_fanout39_out3, %transport_fanout39_out4, %transport_fanout39_out5, %transport_fanout39_out6, %transport_fanout39_out7, %transport_fanout39_out8, %transport_fanout39_out9, %transport_fanout39_out10, %transport_fanout39_out11, %transport_fanout39_out12, %transport_fanout39_out13, %transport_fanout39_out14, %transport_fanout39_out15, %transport_fanout39_out16 = fabric.switch [spatial] %selected
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout40_out0, %transport_fanout40_out1, %transport_fanout40_out2, %transport_fanout40_out3, %transport_fanout40_out4, %transport_fanout40_out5, %transport_fanout40_out6, %transport_fanout40_out7, %transport_fanout40_out8, %transport_fanout40_out9, %transport_fanout40_out10, %transport_fanout40_out11, %transport_fanout40_out12, %transport_fanout40_out13, %transport_fanout40_out14, %transport_fanout40_out15, %transport_fanout40_out16 = fabric.switch [spatial] %addr_shifted
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout41_out0, %transport_fanout41_out1, %transport_fanout41_out2, %transport_fanout41_out3, %transport_fanout41_out4, %transport_fanout41_out5, %transport_fanout41_out6, %transport_fanout41_out7, %transport_fanout41_out8, %transport_fanout41_out9, %transport_fanout41_out10, %transport_fanout41_out11, %transport_fanout41_out12, %transport_fanout41_out13, %transport_fanout41_out14, %transport_fanout41_out15, %transport_fanout41_out16, %transport_fanout41_out17, %transport_fanout41_out18, %transport_fanout41_out19, %transport_fanout41_out20, %transport_fanout41_out21, %transport_fanout41_out22, %transport_fanout41_out23, %transport_fanout41_out24, %transport_fanout41_out25, %transport_fanout41_out26 = fabric.switch [spatial] %idx
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout42_out0, %transport_fanout42_out1, %transport_fanout42_out2, %transport_fanout42_out3, %transport_fanout42_out4, %transport_fanout42_out5, %transport_fanout42_out6, %transport_fanout42_out7, %transport_fanout42_out8, %transport_fanout42_out9, %transport_fanout42_out10, %transport_fanout42_out11 = fabric.switch [spatial] %i32d
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout43_out0, %transport_fanout43_out1, %transport_fanout43_out2, %transport_fanout43_out3, %transport_fanout43_out4, %transport_fanout43_out5, %transport_fanout43_out6, %transport_fanout43_out7, %transport_fanout43_out8, %transport_fanout43_out9, %transport_fanout43_out10, %transport_fanout43_out11, %transport_fanout43_out12, %transport_fanout43_out13, %transport_fanout43_out14, %transport_fanout43_out15, %transport_fanout43_out16, %transport_fanout43_out17, %transport_fanout43_out18, %transport_fanout43_out19, %transport_fanout43_out20, %transport_fanout43_out21 = fabric.switch [spatial] %addr_shift_const
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout44_out0, %transport_fanout44_out1, %transport_fanout44_out2, %transport_fanout44_out3, %transport_fanout44_out4, %transport_fanout44_out5, %transport_fanout44_out6, %transport_fanout44_out7, %transport_fanout44_out8, %transport_fanout44_out9, %transport_fanout44_out10, %transport_fanout44_out11, %transport_fanout44_out12, %transport_fanout44_out13, %transport_fanout44_out14, %transport_fanout44_out15, %transport_fanout44_out16, %transport_fanout44_out17, %transport_fanout44_out18 = fabric.switch [spatial] %addr_aux_const
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout45_out0, %transport_fanout45_out1, %transport_fanout45_out2, %transport_fanout45_out3, %transport_fanout45_out4, %transport_fanout45_out5, %transport_fanout45_out6, %transport_fanout45_out7, %transport_fanout45_out8, %transport_fanout45_out9, %transport_fanout45_out10, %transport_fanout45_out11, %transport_fanout45_out12, %transport_fanout45_out13, %transport_fanout45_out14, %transport_fanout45_out15, %transport_fanout45_out16, %transport_fanout45_out17, %transport_fanout45_out18, %transport_fanout45_out19 = fabric.switch [spatial] %addr_bias_const
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout46_out0, %transport_fanout46_out1, %transport_fanout46_out2, %transport_fanout46_out3, %transport_fanout46_out4, %transport_fanout46_out5, %transport_fanout46_out6, %transport_fanout46_out7, %transport_fanout46_out8, %transport_fanout46_out9, %transport_fanout46_out10, %transport_fanout46_out11, %transport_fanout46_out12, %transport_fanout46_out13, %transport_fanout46_out14, %transport_fanout46_out15, %transport_fanout46_out16, %transport_fanout46_out17, %transport_fanout46_out18, %transport_fanout46_out19, %transport_fanout46_out20, %transport_fanout46_out21, %transport_fanout46_out22, %transport_fanout46_out23, %transport_fanout46_out24, %transport_fanout46_out25, %transport_fanout46_out26, %transport_fanout46_out27, %transport_fanout46_out28 = fabric.switch [spatial] %logic_masked
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout47_out0, %transport_fanout47_out1, %transport_fanout47_out2, %transport_fanout47_out3, %transport_fanout47_out4, %transport_fanout47_out5, %transport_fanout47_out6, %transport_fanout47_out7, %transport_fanout47_out8, %transport_fanout47_out9, %transport_fanout47_out10, %transport_fanout47_out11, %transport_fanout47_out12, %transport_fanout47_out13, %transport_fanout47_out14, %transport_fanout47_out15, %transport_fanout47_out16, %transport_fanout47_out17, %transport_fanout47_out18, %transport_fanout47_out19 = fabric.switch [spatial] %addr_masked
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout48_out0, %transport_fanout48_out1, %transport_fanout48_out2, %transport_fanout48_out3, %transport_fanout48_out4, %transport_fanout48_out5, %transport_fanout48_out6, %transport_fanout48_out7, %transport_fanout48_out8, %transport_fanout48_out9, %transport_fanout48_out10, %transport_fanout48_out11, %transport_fanout48_out12, %transport_fanout48_out13, %transport_fanout48_out14, %transport_fanout48_out15 = fabric.switch [spatial] %aux_masked
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout49_out0, %transport_fanout49_out1, %transport_fanout49_out2, %transport_fanout49_out3, %transport_fanout49_out4, %transport_fanout49_out5, %transport_fanout49_out6, %transport_fanout49_out7, %transport_fanout49_out8, %transport_fanout49_out9, %transport_fanout49_out10, %transport_fanout49_out11, %transport_fanout49_out12, %transport_fanout49_out13, %transport_fanout49_out14 = fabric.switch [spatial] %squared_data
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout50_out0, %transport_fanout50_out1, %transport_fanout50_out2, %transport_fanout50_out3, %transport_fanout50_out4, %transport_fanout50_out5, %transport_fanout50_out6, %transport_fanout50_out7, %transport_fanout50_out8, %transport_fanout50_out9, %transport_fanout50_out10, %transport_fanout50_out11, %transport_fanout50_out12, %transport_fanout50_out13, %transport_fanout50_out14, %transport_fanout50_out15, %transport_fanout50_out16, %transport_fanout50_out17, %transport_fanout50_out18 = fabric.switch [spatial] %int_product
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout51_out0, %transport_fanout51_out1, %transport_fanout51_out2, %transport_fanout51_out3, %transport_fanout51_out4, %transport_fanout51_out5, %transport_fanout51_out6, %transport_fanout51_out7, %transport_fanout51_out8, %transport_fanout51_out9, %transport_fanout51_out10, %transport_fanout51_out11 = fabric.switch [spatial] %int_product_aux
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout52_out0, %transport_fanout52_out1, %transport_fanout52_out2, %transport_fanout52_out3, %transport_fanout52_out4, %transport_fanout52_out5, %transport_fanout52_out6, %transport_fanout52_out7, %transport_fanout52_out8, %transport_fanout52_out9, %transport_fanout52_out10, %transport_fanout52_out11, %transport_fanout52_out12, %transport_fanout52_out13, %transport_fanout52_out14, %transport_fanout52_out15, %transport_fanout52_out16, %transport_fanout52_out17 = fabric.switch [spatial] %aux_invariant0
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout53_out0, %transport_fanout53_out1, %transport_fanout53_out2, %transport_fanout53_out3, %transport_fanout53_out4, %transport_fanout53_out5, %transport_fanout53_out6, %transport_fanout53_out7, %transport_fanout53_out8, %transport_fanout53_out9, %transport_fanout53_out10, %transport_fanout53_out11, %transport_fanout53_out12, %transport_fanout53_out13, %transport_fanout53_out14, %transport_fanout53_out15, %transport_fanout53_out16 = fabric.switch [spatial] %aux_invariant1
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout54_out0, %transport_fanout54_out1, %transport_fanout54_out2, %transport_fanout54_out3, %transport_fanout54_out4, %transport_fanout54_out5, %transport_fanout54_out6, %transport_fanout54_out7, %transport_fanout54_out8, %transport_fanout54_out9, %transport_fanout54_out10, %transport_fanout54_out11, %transport_fanout54_out12, %transport_fanout54_out13, %transport_fanout54_out14, %transport_fanout54_out15, %transport_fanout54_out16 = fabric.switch [spatial] %aux_invariant2
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout55_out0, %transport_fanout55_out1, %transport_fanout55_out2, %transport_fanout55_out3, %transport_fanout55_out4, %transport_fanout55_out5, %transport_fanout55_out6, %transport_fanout55_out7, %transport_fanout55_out8, %transport_fanout55_out9, %transport_fanout55_out10, %transport_fanout55_out11, %transport_fanout55_out12, %transport_fanout55_out13, %transport_fanout55_out14, %transport_fanout55_out15, %transport_fanout55_out16, %transport_fanout55_out17, %transport_fanout55_out18, %transport_fanout55_out19, %transport_fanout55_out20, %transport_fanout55_out21, %transport_fanout55_out22, %transport_fanout55_out23, %transport_fanout55_out24, %transport_fanout55_out25 = fabric.switch [spatial] %cast0_result
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout56_out0, %transport_fanout56_out1, %transport_fanout56_out2, %transport_fanout56_out3, %transport_fanout56_out4, %transport_fanout56_out5, %transport_fanout56_out6, %transport_fanout56_out7, %transport_fanout56_out8, %transport_fanout56_out9, %transport_fanout56_out10, %transport_fanout56_out11, %transport_fanout56_out12, %transport_fanout56_out13, %transport_fanout56_out14, %transport_fanout56_out15, %transport_fanout56_out16, %transport_fanout56_out17, %transport_fanout56_out18, %transport_fanout56_out19, %transport_fanout56_out20, %transport_fanout56_out21, %transport_fanout56_out22, %transport_fanout56_out23, %transport_fanout56_out24 = fabric.switch [spatial] %cast1_result
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout57_out0, %transport_fanout57_out1, %transport_fanout57_out2, %transport_fanout57_out3, %transport_fanout57_out4, %transport_fanout57_out5, %transport_fanout57_out6, %transport_fanout57_out7, %transport_fanout57_out8, %transport_fanout57_out9, %transport_fanout57_out10, %transport_fanout57_out11, %transport_fanout57_out12, %transport_fanout57_out13, %transport_fanout57_out14, %transport_fanout57_out15, %transport_fanout57_out16, %transport_fanout57_out17, %transport_fanout57_out18, %transport_fanout57_out19, %transport_fanout57_out20, %transport_fanout57_out21 = fabric.switch [spatial] %cast2_result
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout58_out0, %transport_fanout58_out1, %transport_fanout58_out2, %transport_fanout58_out3, %transport_fanout58_out4, %transport_fanout58_out5, %transport_fanout58_out6, %transport_fanout58_out7, %transport_fanout58_out8, %transport_fanout58_out9, %transport_fanout58_out10, %transport_fanout58_out11, %transport_fanout58_out12, %transport_fanout58_out13, %transport_fanout58_out14, %transport_fanout58_out15, %transport_fanout58_out16, %transport_fanout58_out17, %transport_fanout58_out18, %transport_fanout58_out19, %transport_fanout58_out20 = fabric.switch [spatial] %cast3_result
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout59_out0, %transport_fanout59_out1, %transport_fanout59_out2, %transport_fanout59_out3, %transport_fanout59_out4, %transport_fanout59_out5, %transport_fanout59_out6, %transport_fanout59_out7, %transport_fanout59_out8, %transport_fanout59_out9, %transport_fanout59_out10, %transport_fanout59_out11, %transport_fanout59_out12, %transport_fanout59_out13, %transport_fanout59_out14 = fabric.switch [spatial] %int_extui
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout60_out0, %transport_fanout60_out1, %transport_fanout60_out2, %transport_fanout60_out3 = fabric.switch [spatial] %wide_truncated
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout61_out0, %transport_fanout61_out1, %transport_fanout61_out2 = fabric.switch [spatial] %wide_truncated_aux
         [{connectivity_table = ["1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout62_out0, %transport_fanout62_out1, %transport_fanout62_out2, %transport_fanout62_out3, %transport_fanout62_out4, %transport_fanout62_out5 = fabric.switch [spatial] %int_rem
         [{connectivity_table = ["1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout63_out0, %transport_fanout63_out1, %transport_fanout63_out2, %transport_fanout63_out3, %transport_fanout63_out4, %transport_fanout63_out5, %transport_fanout63_out6, %transport_fanout63_out7, %transport_fanout63_out8, %transport_fanout63_out9, %transport_fanout63_out10, %transport_fanout63_out11, %transport_fanout63_out12, %transport_fanout63_out13, %transport_fanout63_out14, %transport_fanout63_out15, %transport_fanout63_out16, %transport_fanout63_out17, %transport_fanout63_out18, %transport_fanout63_out19, %transport_fanout63_out20, %transport_fanout63_out21, %transport_fanout63_out22 = fabric.switch [spatial] %aux_idx
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout64_out0, %transport_fanout64_out1, %transport_fanout64_out2, %transport_fanout64_out3, %transport_fanout64_out4, %transport_fanout64_out5, %transport_fanout64_out6, %transport_fanout64_out7, %transport_fanout64_out8, %transport_fanout64_out9, %transport_fanout64_out10, %transport_fanout64_out11, %transport_fanout64_out12, %transport_fanout64_out13, %transport_fanout64_out14, %transport_fanout64_out15, %transport_fanout64_out16, %transport_fanout64_out17, %transport_fanout64_out18 = fabric.switch [spatial] %aux_active_idx
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout65_out0, %transport_fanout65_out1, %transport_fanout65_out2, %transport_fanout65_out3, %transport_fanout65_out4, %transport_fanout65_out5 = fabric.switch [spatial] %int_div0
         [{connectivity_table = ["1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout66_out0, %transport_fanout66_out1, %transport_fanout66_out2, %transport_fanout66_out3, %transport_fanout66_out4, %transport_fanout66_out5 = fabric.switch [spatial] %int_div1
         [{connectivity_table = ["1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout67_out0, %transport_fanout67_out1, %transport_fanout67_out2, %transport_fanout67_out3, %transport_fanout67_out4, %transport_fanout67_out5, %transport_fanout67_out6, %transport_fanout67_out7, %transport_fanout67_out8, %transport_fanout67_out9, %transport_fanout67_out10, %transport_fanout67_out11, %transport_fanout67_out12, %transport_fanout67_out13, %transport_fanout67_out14, %transport_fanout67_out15, %transport_fanout67_out16, %transport_fanout67_out17, %transport_fanout67_out18, %transport_fanout67_out19, %transport_fanout67_out20, %transport_fanout67_out21, %transport_fanout67_out22, %transport_fanout67_out23, %transport_fanout67_out24, %transport_fanout67_out25, %transport_fanout67_out26, %transport_fanout67_out27, %transport_fanout67_out28, %transport_fanout67_out29, %transport_fanout67_out30 = fabric.switch [spatial] %int_sum
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout68_out0, %transport_fanout68_out1, %transport_fanout68_out2, %transport_fanout68_out3, %transport_fanout68_out4, %transport_fanout68_out5, %transport_fanout68_out6, %transport_fanout68_out7, %transport_fanout68_out8, %transport_fanout68_out9, %transport_fanout68_out10, %transport_fanout68_out11, %transport_fanout68_out12, %transport_fanout68_out13, %transport_fanout68_out14, %transport_fanout68_out15, %transport_fanout68_out16, %transport_fanout68_out17, %transport_fanout68_out18, %transport_fanout68_out19, %transport_fanout68_out20, %transport_fanout68_out21, %transport_fanout68_out22, %transport_fanout68_out23, %transport_fanout68_out24, %transport_fanout68_out25, %transport_fanout68_out26 = fabric.switch [spatial] %addr_sum
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout69_out0, %transport_fanout69_out1, %transport_fanout69_out2, %transport_fanout69_out3, %transport_fanout69_out4, %transport_fanout69_out5, %transport_fanout69_out6 = fabric.switch [spatial] %rotated
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout70_out0, %transport_fanout70_out1, %transport_fanout70_out2, %transport_fanout70_out3, %transport_fanout70_out4, %transport_fanout70_out5, %transport_fanout70_out6, %transport_fanout70_out7, %transport_fanout70_out8, %transport_fanout70_out9 = fabric.switch [spatial] %packed_sat
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout71_out0, %transport_fanout71_out1, %transport_fanout71_out2, %transport_fanout71_out3, %transport_fanout71_out4, %transport_fanout71_out5, %transport_fanout71_out6, %transport_fanout71_out7, %transport_fanout71_out8, %transport_fanout71_out9, %transport_fanout71_out10 = fabric.switch [spatial] %cmpf_pred
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout72_out0, %transport_fanout72_out1, %transport_fanout72_out2, %transport_fanout72_out3, %transport_fanout72_out4, %transport_fanout72_out5, %transport_fanout72_out6 = fabric.switch [spatial] %cmpi64_pred_aux_narrow
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout73_out0, %transport_fanout73_out1 = fabric.switch [spatial] %aux_rwc
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout74_out0, %transport_fanout74_out1, %transport_fanout74_out2, %transport_fanout74_out3, %transport_fanout74_out4, %transport_fanout74_out5, %transport_fanout74_out6, %transport_fanout74_out7, %transport_fanout74_out8, %transport_fanout74_out9 = fabric.switch [spatial] %mac_result
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout75_out0, %transport_fanout75_out1, %transport_fanout75_out2, %transport_fanout75_out3, %transport_fanout75_out4, %transport_fanout75_out5, %transport_fanout75_out6, %transport_fanout75_out7, %transport_fanout75_out8, %transport_fanout75_out9, %transport_fanout75_out10 = fabric.switch [spatial] %mac_result1
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout76_out0, %transport_fanout76_out1, %transport_fanout76_out2, %transport_fanout76_out3, %transport_fanout76_out4, %transport_fanout76_out5, %transport_fanout76_out6, %transport_fanout76_out7 = fabric.switch [spatial] %mac_result2
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout77_out0, %transport_fanout77_out1, %transport_fanout77_out2, %transport_fanout77_out3, %transport_fanout77_out4, %transport_fanout77_out5, %transport_fanout77_out6 = fabric.switch [spatial] %mac_result3
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout78_out0, %transport_fanout78_out1, %transport_fanout78_out2, %transport_fanout78_out3, %transport_fanout78_out4, %transport_fanout78_out5, %transport_fanout78_out6, %transport_fanout78_out7, %transport_fanout78_out8, %transport_fanout78_out9, %transport_fanout78_out10 = fabric.switch [spatial] %fp_running
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout79_out0, %transport_fanout79_out1, %transport_fanout79_out2, %transport_fanout79_out3, %transport_fanout79_out4, %transport_fanout79_out5, %transport_fanout79_out6, %transport_fanout79_out7, %transport_fanout79_out8, %transport_fanout79_out9, %transport_fanout79_out10, %transport_fanout79_out11, %transport_fanout79_out12 = fabric.switch [spatial] %fp_running_aux
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout80_out0, %transport_fanout80_out1, %transport_fanout80_out2, %transport_fanout80_out3, %transport_fanout80_out4, %transport_fanout80_out5, %transport_fanout80_out6, %transport_fanout80_out7, %transport_fanout80_out8, %transport_fanout80_out9, %transport_fanout80_out10, %transport_fanout80_out11, %transport_fanout80_out12 = fabric.switch [spatial] %scaled_reduction
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout81_out0, %transport_fanout81_out1, %transport_fanout81_out2, %transport_fanout81_out3, %transport_fanout81_out4 = fabric.switch [spatial] %control_demux_false
         [{connectivity_table = ["1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout82_out0, %transport_fanout82_out1, %transport_fanout82_out2, %transport_fanout82_out3, %transport_fanout82_out4 = fabric.switch [spatial] %compute_demux_true
         [{connectivity_table = ["1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout83_out0, %transport_fanout83_out1, %transport_fanout83_out2, %transport_fanout83_out3, %transport_fanout83_out4, %transport_fanout83_out5, %transport_fanout83_out6, %transport_fanout83_out7, %transport_fanout83_out8 = fabric.switch [spatial] %control_token_demux_false_token
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout84_out0, %transport_fanout84_out1, %transport_fanout84_out2, %transport_fanout84_out3, %transport_fanout84_out4, %transport_fanout84_out5, %transport_fanout84_out6 = fabric.switch [spatial] %control_token_demux_true_token
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout85_out0, %transport_fanout85_out1, %transport_fanout85_out2, %transport_fanout85_out3 = fabric.switch [spatial] %addr_extra_const0
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout86_out0, %transport_fanout86_out1, %transport_fanout86_out2, %transport_fanout86_out3 = fabric.switch [spatial] %addr_extra_const1
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout87_out0, %transport_fanout87_out1, %transport_fanout87_out2, %transport_fanout87_out3, %transport_fanout87_out4, %transport_fanout87_out5, %transport_fanout87_out6, %transport_fanout87_out7 = fabric.switch [spatial] %wide_index_cast0_narrow
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout88_out0, %transport_fanout88_out1, %transport_fanout88_out2, %transport_fanout88_out3, %transport_fanout88_out4, %transport_fanout88_out5, %transport_fanout88_out6, %transport_fanout88_out7 = fabric.switch [spatial] %wide_index_cast1_narrow
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout89_out0, %transport_fanout89_out1, %transport_fanout89_out2, %transport_fanout89_out3, %transport_fanout89_out4, %transport_fanout89_out5, %transport_fanout89_out6, %transport_fanout89_out7 = fabric.switch [spatial] %uint_rem
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout90_out0, %transport_fanout90_out1, %transport_fanout90_out2 = fabric.switch [spatial] %signed_min
         [{connectivity_table = ["1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout91_out0, %transport_fanout91_out1, %transport_fanout91_out2 = fabric.switch [spatial] %signed_max
         [{connectivity_table = ["1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout92_out0, %transport_fanout92_out1, %transport_fanout92_out2, %transport_fanout92_out3, %transport_fanout92_out4, %transport_fanout92_out5, %transport_fanout92_out6, %transport_fanout92_out7, %transport_fanout92_out8, %transport_fanout92_out9, %transport_fanout92_out10, %transport_fanout92_out11, %transport_fanout92_out12, %transport_fanout92_out13, %transport_fanout92_out14, %transport_fanout92_out15, %transport_fanout92_out16, %transport_fanout92_out17, %transport_fanout92_out18 = fabric.switch [spatial] %wide_zext1
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout93_out0, %transport_fanout93_out1, %transport_fanout93_out2, %transport_fanout93_out3, %transport_fanout93_out4, %transport_fanout93_out5, %transport_fanout93_out6, %transport_fanout93_out7, %transport_fanout93_out8, %transport_fanout93_out9, %transport_fanout93_out10, %transport_fanout93_out11, %transport_fanout93_out12, %transport_fanout93_out13, %transport_fanout93_out14, %transport_fanout93_out15, %transport_fanout93_out16, %transport_fanout93_out17, %transport_fanout93_out18 = fabric.switch [spatial] %wide_zext0
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout94_out0, %transport_fanout94_out1, %transport_fanout94_out2, %transport_fanout94_out3, %transport_fanout94_out4, %transport_fanout94_out5, %transport_fanout94_out6, %transport_fanout94_out7, %transport_fanout94_out8, %transport_fanout94_out9, %transport_fanout94_out10, %transport_fanout94_out11, %transport_fanout94_out12, %transport_fanout94_out13 = fabric.switch [spatial] %wide_product
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout95_out0, %transport_fanout95_out1, %transport_fanout95_out2, %transport_fanout95_out3, %transport_fanout95_out4, %transport_fanout95_out5, %transport_fanout95_out6, %transport_fanout95_out7, %transport_fanout95_out8, %transport_fanout95_out9, %transport_fanout95_out10 = fabric.switch [spatial] %wide_shifted
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout96_out0, %transport_fanout96_out1, %transport_fanout96_out2, %transport_fanout96_out3, %transport_fanout96_out4, %transport_fanout96_out5, %transport_fanout96_out6, %transport_fanout96_out7, %transport_fanout96_out8, %transport_fanout96_out9, %transport_fanout96_out10, %transport_fanout96_out11 = fabric.switch [spatial] %wide_signed_quotient
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout97_out0, %transport_fanout97_out1, %transport_fanout97_out2, %transport_fanout97_out3, %transport_fanout97_out4, %transport_fanout97_out5, %transport_fanout97_out6, %transport_fanout97_out7, %transport_fanout97_out8, %transport_fanout97_out9, %transport_fanout97_out10, %transport_fanout97_out11 = fabric.switch [spatial] %wide_remainder
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout98_out0, %transport_fanout98_out1, %transport_fanout98_out2, %transport_fanout98_out3, %transport_fanout98_out4, %transport_fanout98_out5, %transport_fanout98_out6, %transport_fanout98_out7 = fabric.switch [spatial] %wide_sum
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout99_out0, %transport_fanout99_out1, %transport_fanout99_out2, %transport_fanout99_out3, %transport_fanout99_out4, %transport_fanout99_out5 = fabric.switch [spatial] %wide_sum_aux
         [{connectivity_table = ["1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout100_out0, %transport_fanout100_out1 = fabric.switch [spatial] %wide_pred_extui
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>)
  %transport_fanout101_out0, %transport_fanout101_out1, %transport_fanout101_out2, %transport_fanout101_out3, %transport_fanout101_out4, %transport_fanout101_out5, %transport_fanout101_out6, %transport_fanout101_out7, %transport_fanout101_out8, %transport_fanout101_out9, %transport_fanout101_out10, %transport_fanout101_out11, %transport_fanout101_out12, %transport_fanout101_out13, %transport_fanout101_out14, %transport_fanout101_out15 = fabric.switch [spatial] %data2
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout102_out0, %transport_fanout102_out1, %transport_fanout102_out2, %transport_fanout102_out3, %transport_fanout102_out4, %transport_fanout102_out5, %transport_fanout102_out6, %transport_fanout102_out7, %transport_fanout102_out8, %transport_fanout102_out9, %transport_fanout102_out10, %transport_fanout102_out11, %transport_fanout102_out12, %transport_fanout102_out13, %transport_fanout102_out14, %transport_fanout102_out15 = fabric.switch [spatial] %data3
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout103_out0, %transport_fanout103_out1, %transport_fanout103_out2, %transport_fanout103_out3, %transport_fanout103_out4, %transport_fanout103_out5, %transport_fanout103_out6, %transport_fanout103_out7, %transport_fanout103_out8, %transport_fanout103_out9, %transport_fanout103_out10, %transport_fanout103_out11, %transport_fanout103_out12, %transport_fanout103_out13, %transport_fanout103_out14, %transport_fanout103_out15, %transport_fanout103_out16 = fabric.switch [spatial] %data4
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout104_out0, %transport_fanout104_out1, %transport_fanout104_out2, %transport_fanout104_out3, %transport_fanout104_out4, %transport_fanout104_out5, %transport_fanout104_out6, %transport_fanout104_out7, %transport_fanout104_out8, %transport_fanout104_out9, %transport_fanout104_out10, %transport_fanout104_out11, %transport_fanout104_out12, %transport_fanout104_out13, %transport_fanout104_out14, %transport_fanout104_out15 = fabric.switch [spatial] %data5
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout105_out0, %transport_fanout105_out1, %transport_fanout105_out2, %transport_fanout105_out3, %transport_fanout105_out4, %transport_fanout105_out5, %transport_fanout105_out6, %transport_fanout105_out7, %transport_fanout105_out8 = fabric.switch [spatial] %fp_diff
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout106_out0, %transport_fanout106_out1, %transport_fanout106_out2, %transport_fanout106_out3, %transport_fanout106_out4, %transport_fanout106_out5, %transport_fanout106_out6, %transport_fanout106_out7, %transport_fanout106_out8 = fabric.switch [spatial] %fp_diff_aux
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout107_out0, %transport_fanout107_out1, %transport_fanout107_out2, %transport_fanout107_out3, %transport_fanout107_out4 = fabric.switch [spatial] %int_or
         [{connectivity_table = ["1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout108_out0, %transport_fanout108_out1 = fabric.switch [spatial] %abs_data
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout109_out0, %transport_fanout109_out1, %transport_fanout109_out2, %transport_fanout109_out3, %transport_fanout109_out4 = fabric.switch [spatial] %scaled_reduction_aux
         [{connectivity_table = ["1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout110_out0, %transport_fanout110_out1, %transport_fanout110_out2, %transport_fanout110_out3 = fabric.switch [spatial] %compute_demux_false
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout111_out0, %transport_fanout111_out1, %transport_fanout111_out2, %transport_fanout111_out3, %transport_fanout111_out4, %transport_fanout111_out5, %transport_fanout111_out6, %transport_fanout111_out7 = fabric.switch [spatial] %fp_negated
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout112_out0, %transport_fanout112_out1, %transport_fanout112_out2, %transport_fanout112_out3, %transport_fanout112_out4, %transport_fanout112_out5, %transport_fanout112_out6, %transport_fanout112_out7, %transport_fanout112_out8 = fabric.switch [spatial] %done1
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout113_out0, %transport_fanout113_out1, %transport_fanout113_out2, %transport_fanout113_out3, %transport_fanout113_out4, %transport_fanout113_out5, %transport_fanout113_out6 = fabric.switch [spatial] %control_token_muxed_token
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout114_out0, %transport_fanout114_out1, %transport_fanout114_out2, %transport_fanout114_out3, %transport_fanout114_out4, %transport_fanout114_out5, %transport_fanout114_out6, %transport_fanout114_out7, %transport_fanout114_out8 = fabric.switch [spatial] %done2
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout115_out0, %transport_fanout115_out1, %transport_fanout115_out2, %transport_fanout115_out3, %transport_fanout115_out4 = fabric.switch [spatial] %store_done1
         [{connectivity_table = ["1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout116_out0, %transport_fanout116_out1, %transport_fanout116_out2, %transport_fanout116_out3, %transport_fanout116_out4, %transport_fanout116_out5, %transport_fanout116_out6, %transport_fanout116_out7 = fabric.switch [spatial] %done3
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout117_out0, %transport_fanout117_out1, %transport_fanout117_out2, %transport_fanout117_out3, %transport_fanout117_out4, %transport_fanout117_out5, %transport_fanout117_out6, %transport_fanout117_out7, %transport_fanout117_out8 = fabric.switch [spatial] %done5
         [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout118_out0, %transport_fanout118_out1, %transport_fanout118_out2, %transport_fanout118_out3 = fabric.switch [spatial] %vector_sync_done
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout119_out0, %transport_fanout119_out1, %transport_fanout119_out2, %transport_fanout119_out3 = fabric.switch [spatial] %sync_done
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %transport_fanout120_out0, %transport_fanout120_out1, %transport_fanout120_out2 = fabric.switch [spatial] %leading_zero_count
         [{connectivity_table = ["1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout121_out0, %transport_fanout121_out1, %transport_fanout121_out2, %transport_fanout121_out3 = fabric.switch [spatial] %control_demux_true
         [{connectivity_table = ["1", "1", "1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %transport_fanout122_out0, %transport_fanout122_out1 = fabric.switch [spatial] %fp_div
         [{connectivity_table = ["1", "1"]}]
         : (!fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}
