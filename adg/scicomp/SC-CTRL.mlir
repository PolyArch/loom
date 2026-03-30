module {
fabric.module @SC-CTRL_core() -> () attributes {loom.decomposable = false, loom.extmem_ld_ports = 2, loom.extmem_st_ports = 2, loom.fp_add_units = 2, loom.fp_div_units = 0, loom.fp_mul_units = 2, loom.has_branch = true, loom.has_fma = false, loom.has_fp_min = false, loom.has_indirect_load = false, loom.has_rsqrt = false, loom.has_scatter_store = true, loom.int_alu_units = 6, loom.int_mul_units = 2, loom.operand_buffer_size = 4, loom.routing_topology = "MESH", loom.scicomp_khg_type = "SC-CTRL", loom.spm_ld_ports = 2, loom.spm_st_ports = 2, loom.sub_lane_bits = 0} {
  fabric.temporal_pe @SC-CTRL_core_tpe(%in0: !fabric.bits<32>, %in1: !fabric.bits<32>, %in2: !fabric.bits<32>, %in3: !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) [num_register = 8, num_instruction = 16, reg_fifo_depth = 1, enable_share_operand_buffer = true, operand_buffer_size = 4] {
    fabric.function_unit @SC-CTRL_core_const_int(%arg0: none) -> (i32) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0 : i32} : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_const_index(%arg0: none) -> (index) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0 : index} : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-CTRL_core_const_float(%arg0: none) -> (f32) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0.000000e+00 : f32} : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_index_to_int(%arg0: index) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : index to i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_int_to_index(%arg0: i32) -> (index) [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : i32 to index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-CTRL_core_stream(%arg0: index, %arg1: index, %arg2: index) -> (index, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.stream %arg0, %arg1, %arg2 {step_op = "+=", cont_cond = "<"} : (index, index, index) -> (index, i1)
      fabric.yield %0, %1 : index, i1
    }
    fabric.function_unit @SC-CTRL_core_mux_int(%arg0: index, %arg1: i32, %arg2: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_mux_none(%arg0: index, %arg1: none, %arg2: none) -> (none) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-CTRL_core_mux_index(%arg0: index, %arg1: index, %arg2: index) -> (index) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-CTRL_core_join(%arg0: none, %arg1: none, %arg2: none, %arg3: none) -> (none) [latency = 1, interval = 1] {
      %0 = handshake.join %arg0, %arg1, %arg2, %arg3 : none, none, none, none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-CTRL_core_gate_int(%arg0: i32, %arg1: i1) -> (i32, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : i32, i1 -> i32, i1
      fabric.yield %0, %1 : i32, i1
    }
    fabric.function_unit @SC-CTRL_core_gate_index(%arg0: index, %arg1: i1) -> (index, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : index, i1 -> index, i1
      fabric.yield %0, %1 : index, i1
    }
    fabric.function_unit @SC-CTRL_core_gate_float(%arg0: f32, %arg1: i1) -> (f32, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : f32, i1 -> f32, i1
      fabric.yield %0, %1 : f32, i1
    }
    fabric.function_unit @SC-CTRL_core_gate_i1(%arg0: i1, %arg1: i1) -> (i1, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : i1, i1 -> i1, i1
      fabric.yield %0, %1 : i1, i1
    }
    fabric.function_unit @SC-CTRL_core_carry_int(%arg0: i1, %arg1: i32, %arg2: i32) -> (i32) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, i32, i32 -> i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_carry_none(%arg0: i1, %arg1: none, %arg2: none) -> (none) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, none, none -> none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-CTRL_core_carry_float(%arg0: i1, %arg1: f32, %arg2: f32) -> (f32) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, f32, f32 -> f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_cond_br_int(%arg0: i1, %arg1: i32) -> (i32, i32) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : i32
      fabric.yield %0, %1 : i32, i32
    }
    fabric.function_unit @SC-CTRL_core_cond_br_none(%arg0: i1, %arg1: none) -> (none, none) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : none
      fabric.yield %0, %1 : none, none
    }
    fabric.function_unit @SC-CTRL_core_cond_br_float(%arg0: i1, %arg1: f32) -> (f32, f32) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : f32
      fabric.yield %0, %1 : f32, f32
    }
    fabric.function_unit @SC-CTRL_core_invariant_int(%arg0: i1, %arg1: i32) -> (i32) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, i32 -> i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_invariant_index(%arg0: i1, %arg1: index) -> (index) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, index -> index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-CTRL_core_invariant_float(%arg0: i1, %arg1: f32) -> (f32) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, f32 -> f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_invariant_none(%arg0: i1, %arg1: none) -> (none) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, none -> none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-CTRL_core_invariant_i1(%arg0: i1, %arg1: i1) -> (i1) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, i1 -> i1
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-CTRL_core_load(%arg0: index, %arg1: i32, %arg2: none) -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.function_unit @SC-CTRL_core_store(%arg0: index, %arg1: i32, %arg2: none) -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.store [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.function_unit @SC-CTRL_core_select_int(%arg0: i1, %arg1: i32, %arg2: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_select_index(%arg0: i1, %arg1: index, %arg2: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-CTRL_core_cmpi_int(%arg0: i32, %arg1: i32) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpi slt, %arg0, %arg1 : i32
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-CTRL_core_addi_index(%arg0: index, %arg1: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-CTRL_core_muli_index(%arg0: index, %arg1: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.muli %arg0, %arg1 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-CTRL_core_alu0_addi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu0_subi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu0_andi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu0_ori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu0_xori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu0_shli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu0_shrsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu0_shrui(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_addi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_subi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_andi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_ori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_xori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_shli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_shrsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu1_shrui(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_addi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_subi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_andi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_ori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_xori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_shli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_shrsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu2_shrui(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_addi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_subi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_andi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_ori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_xori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_shli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_shrsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu3_shrui(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_addi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_subi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_andi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_ori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_xori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_shli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_shrsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu4_shrui(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_addi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_subi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_andi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_ori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_xori(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_shli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_shrsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_alu5_shrui(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_mul0_muli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.muli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_mul0_divsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.divsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_mul0_remsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.remsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_mul1_muli(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.muli %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_mul1_divsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.divsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_mul1_remsi(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.remsi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_fp0_addf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp0_subf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp0_mulf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp0_divf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp0_cmpf(%arg0: f32, %arg1: f32) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f32
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-CTRL_core_fp0_select_float(%arg0: i1, %arg1: f32, %arg2: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp0_sitofp(%arg0: i32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i32 to f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp0_fptosi(%arg0: f32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f32 to i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_fp0_negf(%arg0: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp1_addf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp1_subf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp1_mulf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp1_divf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp1_cmpf(%arg0: f32, %arg1: f32) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f32
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-CTRL_core_fp1_select_float(%arg0: i1, %arg1: f32, %arg2: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp1_sitofp(%arg0: i32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i32 to f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp1_fptosi(%arg0: f32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f32 to i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_fp1_negf(%arg0: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp2_addf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp2_subf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp2_mulf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp2_divf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp2_cmpf(%arg0: f32, %arg1: f32) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f32
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-CTRL_core_fp2_select_float(%arg0: i1, %arg1: f32, %arg2: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp2_sitofp(%arg0: i32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i32 to f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp2_fptosi(%arg0: f32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f32 to i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_fp2_negf(%arg0: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp3_addf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp3_subf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp3_mulf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp3_divf(%arg0: f32, %arg1: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp3_cmpf(%arg0: f32, %arg1: f32) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f32
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-CTRL_core_fp3_select_float(%arg0: i1, %arg1: f32, %arg2: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp3_sitofp(%arg0: i32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i32 to f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_fp3_fptosi(%arg0: f32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f32 to i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @SC-CTRL_core_fp3_negf(%arg0: f32) -> (f32) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f32
      fabric.yield %0 : f32
    }
    fabric.function_unit @SC-CTRL_core_scatter_store(%arg0: index, %arg1: i32, %arg2: none) -> (none) [latency = 1, interval = 1] {
      %0, %1 = handshake.store [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : none,                                                  
    }
    fabric.function_unit @SC-CTRL_core_branch(%arg0: i1, %arg1: i32) -> (i32, i32) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : i32
      fabric.yield %0, %1 : i32, i32
    }
    fabric.yield
  }
  %v0:4 = fabric.instance @SC-CTRL_core_tpe(%v56#1, %v8#0, %v1#3, %v7#2) {sym_name = "pe_0_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v1:4 = fabric.instance @SC-CTRL_core_tpe(%v57#1, %v9#0, %v2#3, %v0#2) {sym_name = "pe_0_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v2:4 = fabric.instance @SC-CTRL_core_tpe(%v58#1, %v10#0, %v3#3, %v1#2) {sym_name = "pe_0_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v3:4 = fabric.instance @SC-CTRL_core_tpe(%v59#1, %v11#0, %v4#3, %v2#2) {sym_name = "pe_0_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v4:4 = fabric.instance @SC-CTRL_core_tpe(%v60#1, %v12#0, %v5#3, %v3#2) {sym_name = "pe_0_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v5:4 = fabric.instance @SC-CTRL_core_tpe(%v61#1, %v13#0, %v6#3, %v4#2) {sym_name = "pe_0_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v6:4 = fabric.instance @SC-CTRL_core_tpe(%v62#1, %v14#0, %v7#3, %v5#2) {sym_name = "pe_0_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v7:4 = fabric.instance @SC-CTRL_core_tpe(%v63#1, %v15#0, %v0#3, %v6#2) {sym_name = "pe_0_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v8:4 = fabric.instance @SC-CTRL_core_tpe(%v0#1, %v16#0, %v9#3, %v15#2) {sym_name = "pe_1_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v9:4 = fabric.instance @SC-CTRL_core_tpe(%v1#1, %v17#0, %v10#3, %v8#2) {sym_name = "pe_1_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v10:4 = fabric.instance @SC-CTRL_core_tpe(%v2#1, %v18#0, %v11#3, %v9#2) {sym_name = "pe_1_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v11:4 = fabric.instance @SC-CTRL_core_tpe(%v3#1, %v19#0, %v12#3, %v10#2) {sym_name = "pe_1_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v12:4 = fabric.instance @SC-CTRL_core_tpe(%v4#1, %v20#0, %v13#3, %v11#2) {sym_name = "pe_1_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v13:4 = fabric.instance @SC-CTRL_core_tpe(%v5#1, %v21#0, %v14#3, %v12#2) {sym_name = "pe_1_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v14:4 = fabric.instance @SC-CTRL_core_tpe(%v6#1, %v22#0, %v15#3, %v13#2) {sym_name = "pe_1_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v15:4 = fabric.instance @SC-CTRL_core_tpe(%v7#1, %v23#0, %v8#3, %v14#2) {sym_name = "pe_1_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v16:4 = fabric.instance @SC-CTRL_core_tpe(%v8#1, %v24#0, %v17#3, %v23#2) {sym_name = "pe_2_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v17:4 = fabric.instance @SC-CTRL_core_tpe(%v9#1, %v25#0, %v18#3, %v16#2) {sym_name = "pe_2_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v18:4 = fabric.instance @SC-CTRL_core_tpe(%v10#1, %v26#0, %v19#3, %v17#2) {sym_name = "pe_2_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v19:4 = fabric.instance @SC-CTRL_core_tpe(%v11#1, %v27#0, %v20#3, %v18#2) {sym_name = "pe_2_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v20:4 = fabric.instance @SC-CTRL_core_tpe(%v12#1, %v28#0, %v21#3, %v19#2) {sym_name = "pe_2_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v21:4 = fabric.instance @SC-CTRL_core_tpe(%v13#1, %v29#0, %v22#3, %v20#2) {sym_name = "pe_2_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v22:4 = fabric.instance @SC-CTRL_core_tpe(%v14#1, %v30#0, %v23#3, %v21#2) {sym_name = "pe_2_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v23:4 = fabric.instance @SC-CTRL_core_tpe(%v15#1, %v31#0, %v16#3, %v22#2) {sym_name = "pe_2_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v24:4 = fabric.instance @SC-CTRL_core_tpe(%v16#1, %v32#0, %v25#3, %v31#2) {sym_name = "pe_3_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v25:4 = fabric.instance @SC-CTRL_core_tpe(%v17#1, %v33#0, %v26#3, %v24#2) {sym_name = "pe_3_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v26:4 = fabric.instance @SC-CTRL_core_tpe(%v18#1, %v34#0, %v27#3, %v25#2) {sym_name = "pe_3_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v27:4 = fabric.instance @SC-CTRL_core_tpe(%v19#1, %v35#0, %v28#3, %v26#2) {sym_name = "pe_3_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v28:4 = fabric.instance @SC-CTRL_core_tpe(%v20#1, %v36#0, %v29#3, %v27#2) {sym_name = "pe_3_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v29:4 = fabric.instance @SC-CTRL_core_tpe(%v21#1, %v37#0, %v30#3, %v28#2) {sym_name = "pe_3_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v30:4 = fabric.instance @SC-CTRL_core_tpe(%v22#1, %v38#0, %v31#3, %v29#2) {sym_name = "pe_3_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v31:4 = fabric.instance @SC-CTRL_core_tpe(%v23#1, %v39#0, %v24#3, %v30#2) {sym_name = "pe_3_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v32:4 = fabric.instance @SC-CTRL_core_tpe(%v24#1, %v40#0, %v33#3, %v39#2) {sym_name = "pe_4_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v33:4 = fabric.instance @SC-CTRL_core_tpe(%v25#1, %v41#0, %v34#3, %v32#2) {sym_name = "pe_4_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v34:4 = fabric.instance @SC-CTRL_core_tpe(%v26#1, %v42#0, %v35#3, %v33#2) {sym_name = "pe_4_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v35:4 = fabric.instance @SC-CTRL_core_tpe(%v27#1, %v43#0, %v36#3, %v34#2) {sym_name = "pe_4_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v36:4 = fabric.instance @SC-CTRL_core_tpe(%v28#1, %v44#0, %v37#3, %v35#2) {sym_name = "pe_4_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v37:4 = fabric.instance @SC-CTRL_core_tpe(%v29#1, %v45#0, %v38#3, %v36#2) {sym_name = "pe_4_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v38:4 = fabric.instance @SC-CTRL_core_tpe(%v30#1, %v46#0, %v39#3, %v37#2) {sym_name = "pe_4_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v39:4 = fabric.instance @SC-CTRL_core_tpe(%v31#1, %v47#0, %v32#3, %v38#2) {sym_name = "pe_4_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v40:4 = fabric.instance @SC-CTRL_core_tpe(%v32#1, %v48#0, %v41#3, %v47#2) {sym_name = "pe_5_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v41:4 = fabric.instance @SC-CTRL_core_tpe(%v33#1, %v49#0, %v42#3, %v40#2) {sym_name = "pe_5_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v42:4 = fabric.instance @SC-CTRL_core_tpe(%v34#1, %v50#0, %v43#3, %v41#2) {sym_name = "pe_5_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v43:4 = fabric.instance @SC-CTRL_core_tpe(%v35#1, %v51#0, %v44#3, %v42#2) {sym_name = "pe_5_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v44:4 = fabric.instance @SC-CTRL_core_tpe(%v36#1, %v52#0, %v45#3, %v43#2) {sym_name = "pe_5_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v45:4 = fabric.instance @SC-CTRL_core_tpe(%v37#1, %v53#0, %v46#3, %v44#2) {sym_name = "pe_5_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v46:4 = fabric.instance @SC-CTRL_core_tpe(%v38#1, %v54#0, %v47#3, %v45#2) {sym_name = "pe_5_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v47:4 = fabric.instance @SC-CTRL_core_tpe(%v39#1, %v55#0, %v40#3, %v46#2) {sym_name = "pe_5_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v48:4 = fabric.instance @SC-CTRL_core_tpe(%v40#1, %v56#0, %v49#3, %v55#2) {sym_name = "pe_6_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v49:4 = fabric.instance @SC-CTRL_core_tpe(%v41#1, %v57#0, %v50#3, %v48#2) {sym_name = "pe_6_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v50:4 = fabric.instance @SC-CTRL_core_tpe(%v42#1, %v58#0, %v51#3, %v49#2) {sym_name = "pe_6_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v51:4 = fabric.instance @SC-CTRL_core_tpe(%v43#1, %v59#0, %v52#3, %v50#2) {sym_name = "pe_6_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v52:4 = fabric.instance @SC-CTRL_core_tpe(%v44#1, %v60#0, %v53#3, %v51#2) {sym_name = "pe_6_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v53:4 = fabric.instance @SC-CTRL_core_tpe(%v45#1, %v61#0, %v54#3, %v52#2) {sym_name = "pe_6_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v54:4 = fabric.instance @SC-CTRL_core_tpe(%v46#1, %v62#0, %v55#3, %v53#2) {sym_name = "pe_6_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v55:4 = fabric.instance @SC-CTRL_core_tpe(%v47#1, %v63#0, %v48#3, %v54#2) {sym_name = "pe_6_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v56:4 = fabric.instance @SC-CTRL_core_tpe(%v48#1, %v0#0, %v57#3, %v63#2) {sym_name = "pe_7_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v57:4 = fabric.instance @SC-CTRL_core_tpe(%v49#1, %v1#0, %v58#3, %v56#2) {sym_name = "pe_7_1"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v58:4 = fabric.instance @SC-CTRL_core_tpe(%v50#1, %v2#0, %v59#3, %v57#2) {sym_name = "pe_7_2"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v59:4 = fabric.instance @SC-CTRL_core_tpe(%v51#1, %v3#0, %v60#3, %v58#2) {sym_name = "pe_7_3"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v60:4 = fabric.instance @SC-CTRL_core_tpe(%v52#1, %v4#0, %v61#3, %v59#2) {sym_name = "pe_7_4"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v61:4 = fabric.instance @SC-CTRL_core_tpe(%v53#1, %v5#0, %v62#3, %v60#2) {sym_name = "pe_7_5"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v62:4 = fabric.instance @SC-CTRL_core_tpe(%v54#1, %v6#0, %v63#3, %v61#2) {sym_name = "pe_7_6"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %v63:4 = fabric.instance @SC-CTRL_core_tpe(%v55#1, %v7#0, %v56#3, %v62#2) {sym_name = "pe_7_7"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}
}
// CORE_TYPE_METADATA
// spm_capacity_bytes = 32768
