module {
fabric.module @SC-SPM_core(%mem0: memref<?xi64>, %mem1: memref<?xi64>, %scalar0: !fabric.bits<64>, %scalar1: !fabric.bits<64>, %scalar2: !fabric.bits<64>, %scalar3: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) attributes {loom.decomposable = true, loom.extmem_ld_ports = 4, loom.extmem_st_ports = 2, loom.fp_add_units = 4, loom.fp_div_units = 1, loom.fp_mul_units = 4, loom.has_branch = false, loom.has_fma = true, loom.has_fp_min = false, loom.has_indirect_load = true, loom.has_rsqrt = false, loom.has_scatter_store = false, loom.int_alu_units = 6, loom.int_mul_units = 1, loom.routing_topology = "CHESS", loom.scicomp_khg_type = "SC-SPM", loom.spm_ld_ports = 4, loom.spm_st_ports = 2, loom.sub_lane_bits = 32} {
  fabric.spatial_pe @SC-SPM_core_spe(%in0: !fabric.bits<64>, %in1: !fabric.bits<64>, %in2: !fabric.bits<64>, %in3: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @SC-SPM_core_const_int(%arg0: none) -> (i64) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0 : i64} : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_const_index(%arg0: none) -> (index) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0 : index} : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-SPM_core_const_float(%arg0: none) -> (f64) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0.000000e+00 : f64} : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_index_to_int(%arg0: index) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : index to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_int_to_index(%arg0: i64) -> (index) [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : i64 to index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-SPM_core_stream(%arg0: index, %arg1: index, %arg2: index) -> (index, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.stream %arg0, %arg1, %arg2 {step_op = "+=", cont_cond = "<"} : (index, index, index) -> (index, i1)
      fabric.yield %0, %1 : index, i1
    }
    fabric.function_unit @SC-SPM_core_mux_int(%arg0: index, %arg1: i64, %arg2: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_mux_none(%arg0: index, %arg1: none, %arg2: none) -> (none) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-SPM_core_mux_index(%arg0: index, %arg1: index, %arg2: index) -> (index) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-SPM_core_join(%arg0: none, %arg1: none, %arg2: none, %arg3: none) -> (none) [latency = 1, interval = 1] {
      %0 = handshake.join %arg0, %arg1, %arg2, %arg3 : none, none, none, none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-SPM_core_gate_int(%arg0: i64, %arg1: i1) -> (i64, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : i64, i1 -> i64, i1
      fabric.yield %0, %1 : i64, i1
    }
    fabric.function_unit @SC-SPM_core_gate_index(%arg0: index, %arg1: i1) -> (index, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : index, i1 -> index, i1
      fabric.yield %0, %1 : index, i1
    }
    fabric.function_unit @SC-SPM_core_gate_float(%arg0: f64, %arg1: i1) -> (f64, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : f64, i1 -> f64, i1
      fabric.yield %0, %1 : f64, i1
    }
    fabric.function_unit @SC-SPM_core_gate_i1(%arg0: i1, %arg1: i1) -> (i1, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : i1, i1 -> i1, i1
      fabric.yield %0, %1 : i1, i1
    }
    fabric.function_unit @SC-SPM_core_carry_int(%arg0: i1, %arg1: i64, %arg2: i64) -> (i64) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, i64, i64 -> i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_carry_none(%arg0: i1, %arg1: none, %arg2: none) -> (none) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, none, none -> none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-SPM_core_carry_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, f64, f64 -> f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_cond_br_int(%arg0: i1, %arg1: i64) -> (i64, i64) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : i64
      fabric.yield %0, %1 : i64, i64
    }
    fabric.function_unit @SC-SPM_core_cond_br_none(%arg0: i1, %arg1: none) -> (none, none) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : none
      fabric.yield %0, %1 : none, none
    }
    fabric.function_unit @SC-SPM_core_cond_br_float(%arg0: i1, %arg1: f64) -> (f64, f64) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : f64
      fabric.yield %0, %1 : f64, f64
    }
    fabric.function_unit @SC-SPM_core_invariant_int(%arg0: i1, %arg1: i64) -> (i64) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, i64 -> i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_invariant_index(%arg0: i1, %arg1: index) -> (index) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, index -> index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-SPM_core_invariant_float(%arg0: i1, %arg1: f64) -> (f64) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, f64 -> f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_invariant_none(%arg0: i1, %arg1: none) -> (none) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, none -> none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-SPM_core_invariant_i1(%arg0: i1, %arg1: i1) -> (i1) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, i1 -> i1
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_load(%arg0: index, %arg1: i64, %arg2: none) -> (i64, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i64
      fabric.yield %0, %1 : i64, index
    }
    fabric.function_unit @SC-SPM_core_store(%arg0: index, %arg1: i64, %arg2: none) -> (i64, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.store [%arg0] %arg1, %arg2 : index, i64
      fabric.yield %0, %1 : i64, index
    }
    fabric.function_unit @SC-SPM_core_select_int(%arg0: i1, %arg1: i64, %arg2: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_select_index(%arg0: i1, %arg1: index, %arg2: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-SPM_core_cmpi_int(%arg0: i64, %arg1: i64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpi slt, %arg0, %arg1 : i64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_addi_index(%arg0: index, %arg1: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-SPM_core_muli_index(%arg0: index, %arg1: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.muli %arg0, %arg1 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-SPM_core_alu0_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu0_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu0_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu0_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu0_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu0_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu0_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu0_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu1_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu2_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu3_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu4_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_alu5_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_mul0_muli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.muli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_mul0_divsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.divsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_mul0_remsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.remsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp0_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp0_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp0_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp0_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp0_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp0_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp0_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp0_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp0_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp1_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp1_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp1_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp1_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp1_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp1_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp1_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp1_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp1_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp2_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp2_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp2_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp2_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp2_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp2_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp2_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp2_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp2_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp3_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp3_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp3_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp3_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp3_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp3_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp3_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp3_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp3_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp4_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp4_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp4_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp4_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp4_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp4_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp4_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp4_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp4_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp5_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp5_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp5_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp5_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp5_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp5_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp5_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp5_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp5_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp6_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp6_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp6_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp6_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp6_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp6_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp6_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp6_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp6_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp7_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp7_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp7_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp7_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp7_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp7_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp7_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp7_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp7_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp8_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp8_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp8_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp8_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp8_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-SPM_core_fp8_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp8_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp8_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-SPM_core_fp8_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_fp_fma(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = math.fma %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-SPM_core_indirect_load(%arg0: index, %arg1: i64, %arg2: none) -> (i64, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i64
      fabric.yield %0, %1 : i64, index
    }
    fabric.yield
  }
  fabric.spatial_sw @__chess_sw_10x3_0 [connectivity_table = ["1111111111", "1111111111", "1111111111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_5x5_1 [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_6x3_2 [connectivity_table = ["111111", "111111", "111111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_8x8_3 [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_3x6_4 [connectivity_table = ["111", "111", "111", "111", "111", "111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_3x8_5 [connectivity_table = ["111", "111", "111", "111", "111", "111", "111", "111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v0:3 = fabric.instance @__chess_sw_10x3_0(%v1#0, %v9#1, %v81#0, %v145#0, %v145#1, %v145#2, %scalar0, %scalar1, %scalar2, %scalar3) {sym_name = "sw_0_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v1:5 = fabric.instance @__chess_sw_5x5_1(%v0#0, %v2#0, %v10#2, %v81#1, %v82#0) {sym_name = "sw_0_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v2:5 = fabric.instance @__chess_sw_5x5_1(%v1#1, %v3#0, %v11#2, %v82#1, %v83#0) {sym_name = "sw_0_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v3:5 = fabric.instance @__chess_sw_5x5_1(%v2#1, %v4#0, %v12#2, %v83#1, %v84#0) {sym_name = "sw_0_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v4:5 = fabric.instance @__chess_sw_5x5_1(%v3#1, %v5#0, %v13#2, %v84#1, %v85#0) {sym_name = "sw_0_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v5:5 = fabric.instance @__chess_sw_5x5_1(%v4#1, %v6#0, %v14#2, %v85#1, %v86#0) {sym_name = "sw_0_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v6:5 = fabric.instance @__chess_sw_5x5_1(%v5#1, %v7#0, %v15#2, %v86#1, %v87#0) {sym_name = "sw_0_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v7:5 = fabric.instance @__chess_sw_5x5_1(%v6#1, %v8#0, %v16#2, %v87#1, %v88#0) {sym_name = "sw_0_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v8:3 = fabric.instance @__chess_sw_6x3_2(%v7#1, %v17#1, %v88#1, %v146#0, %v146#1, %v146#2) {sym_name = "sw_0_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v9:5 = fabric.instance @__chess_sw_5x5_1(%v10#0, %v0#1, %v18#1, %v81#2, %v89#0) {sym_name = "sw_1_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v10:8 = fabric.instance @__chess_sw_8x8_3(%v9#0, %v11#0, %v1#2, %v19#2, %v81#3, %v82#2, %v89#1, %v90#0) {sym_name = "sw_1_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v11:8 = fabric.instance @__chess_sw_8x8_3(%v10#1, %v12#0, %v2#2, %v20#2, %v82#3, %v83#2, %v90#1, %v91#0) {sym_name = "sw_1_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v12:8 = fabric.instance @__chess_sw_8x8_3(%v11#1, %v13#0, %v3#2, %v21#2, %v83#3, %v84#2, %v91#1, %v92#0) {sym_name = "sw_1_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v13:8 = fabric.instance @__chess_sw_8x8_3(%v12#1, %v14#0, %v4#2, %v22#2, %v84#3, %v85#2, %v92#1, %v93#0) {sym_name = "sw_1_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v14:8 = fabric.instance @__chess_sw_8x8_3(%v13#1, %v15#0, %v5#2, %v23#2, %v85#3, %v86#2, %v93#1, %v94#0) {sym_name = "sw_1_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v15:8 = fabric.instance @__chess_sw_8x8_3(%v14#1, %v16#0, %v6#2, %v24#2, %v86#3, %v87#2, %v94#1, %v95#0) {sym_name = "sw_1_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v16:8 = fabric.instance @__chess_sw_8x8_3(%v15#1, %v17#0, %v7#2, %v25#2, %v87#3, %v88#2, %v95#1, %v96#0) {sym_name = "sw_1_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v17:5 = fabric.instance @__chess_sw_5x5_1(%v16#1, %v8#1, %v26#1, %v88#3, %v96#1) {sym_name = "sw_1_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v18:5 = fabric.instance @__chess_sw_5x5_1(%v19#0, %v9#2, %v27#1, %v89#2, %v97#0) {sym_name = "sw_2_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v19:8 = fabric.instance @__chess_sw_8x8_3(%v18#0, %v20#0, %v10#3, %v28#2, %v89#3, %v90#2, %v97#1, %v98#0) {sym_name = "sw_2_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v20:8 = fabric.instance @__chess_sw_8x8_3(%v19#1, %v21#0, %v11#3, %v29#2, %v90#3, %v91#2, %v98#1, %v99#0) {sym_name = "sw_2_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v21:8 = fabric.instance @__chess_sw_8x8_3(%v20#1, %v22#0, %v12#3, %v30#2, %v91#3, %v92#2, %v99#1, %v100#0) {sym_name = "sw_2_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v22:8 = fabric.instance @__chess_sw_8x8_3(%v21#1, %v23#0, %v13#3, %v31#2, %v92#3, %v93#2, %v100#1, %v101#0) {sym_name = "sw_2_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v23:8 = fabric.instance @__chess_sw_8x8_3(%v22#1, %v24#0, %v14#3, %v32#2, %v93#3, %v94#2, %v101#1, %v102#0) {sym_name = "sw_2_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v24:8 = fabric.instance @__chess_sw_8x8_3(%v23#1, %v25#0, %v15#3, %v33#2, %v94#3, %v95#2, %v102#1, %v103#0) {sym_name = "sw_2_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v25:8 = fabric.instance @__chess_sw_8x8_3(%v24#1, %v26#0, %v16#3, %v34#2, %v95#3, %v96#2, %v103#1, %v104#0) {sym_name = "sw_2_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v26:5 = fabric.instance @__chess_sw_5x5_1(%v25#1, %v17#2, %v35#1, %v96#3, %v104#1) {sym_name = "sw_2_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v27:5 = fabric.instance @__chess_sw_5x5_1(%v28#0, %v18#2, %v36#1, %v97#2, %v105#0) {sym_name = "sw_3_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v28:8 = fabric.instance @__chess_sw_8x8_3(%v27#0, %v29#0, %v19#3, %v37#2, %v97#3, %v98#2, %v105#1, %v106#0) {sym_name = "sw_3_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v29:8 = fabric.instance @__chess_sw_8x8_3(%v28#1, %v30#0, %v20#3, %v38#2, %v98#3, %v99#2, %v106#1, %v107#0) {sym_name = "sw_3_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v30:8 = fabric.instance @__chess_sw_8x8_3(%v29#1, %v31#0, %v21#3, %v39#2, %v99#3, %v100#2, %v107#1, %v108#0) {sym_name = "sw_3_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v31:8 = fabric.instance @__chess_sw_8x8_3(%v30#1, %v32#0, %v22#3, %v40#2, %v100#3, %v101#2, %v108#1, %v109#0) {sym_name = "sw_3_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v32:8 = fabric.instance @__chess_sw_8x8_3(%v31#1, %v33#0, %v23#3, %v41#2, %v101#3, %v102#2, %v109#1, %v110#0) {sym_name = "sw_3_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v33:8 = fabric.instance @__chess_sw_8x8_3(%v32#1, %v34#0, %v24#3, %v42#2, %v102#3, %v103#2, %v110#1, %v111#0) {sym_name = "sw_3_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v34:8 = fabric.instance @__chess_sw_8x8_3(%v33#1, %v35#0, %v25#3, %v43#2, %v103#3, %v104#2, %v111#1, %v112#0) {sym_name = "sw_3_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v35:5 = fabric.instance @__chess_sw_5x5_1(%v34#1, %v26#2, %v44#1, %v104#3, %v112#1) {sym_name = "sw_3_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v36:5 = fabric.instance @__chess_sw_5x5_1(%v37#0, %v27#2, %v45#1, %v105#2, %v113#0) {sym_name = "sw_4_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v37:8 = fabric.instance @__chess_sw_8x8_3(%v36#0, %v38#0, %v28#3, %v46#2, %v105#3, %v106#2, %v113#1, %v114#0) {sym_name = "sw_4_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v38:8 = fabric.instance @__chess_sw_8x8_3(%v37#1, %v39#0, %v29#3, %v47#2, %v106#3, %v107#2, %v114#1, %v115#0) {sym_name = "sw_4_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v39:8 = fabric.instance @__chess_sw_8x8_3(%v38#1, %v40#0, %v30#3, %v48#2, %v107#3, %v108#2, %v115#1, %v116#0) {sym_name = "sw_4_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v40:8 = fabric.instance @__chess_sw_8x8_3(%v39#1, %v41#0, %v31#3, %v49#2, %v108#3, %v109#2, %v116#1, %v117#0) {sym_name = "sw_4_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v41:8 = fabric.instance @__chess_sw_8x8_3(%v40#1, %v42#0, %v32#3, %v50#2, %v109#3, %v110#2, %v117#1, %v118#0) {sym_name = "sw_4_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v42:8 = fabric.instance @__chess_sw_8x8_3(%v41#1, %v43#0, %v33#3, %v51#2, %v110#3, %v111#2, %v118#1, %v119#0) {sym_name = "sw_4_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v43:8 = fabric.instance @__chess_sw_8x8_3(%v42#1, %v44#0, %v34#3, %v52#2, %v111#3, %v112#2, %v119#1, %v120#0) {sym_name = "sw_4_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v44:5 = fabric.instance @__chess_sw_5x5_1(%v43#1, %v35#2, %v53#1, %v112#3, %v120#1) {sym_name = "sw_4_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v45:5 = fabric.instance @__chess_sw_5x5_1(%v46#0, %v36#2, %v54#1, %v113#2, %v121#0) {sym_name = "sw_5_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v46:8 = fabric.instance @__chess_sw_8x8_3(%v45#0, %v47#0, %v37#3, %v55#2, %v113#3, %v114#2, %v121#1, %v122#0) {sym_name = "sw_5_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v47:8 = fabric.instance @__chess_sw_8x8_3(%v46#1, %v48#0, %v38#3, %v56#2, %v114#3, %v115#2, %v122#1, %v123#0) {sym_name = "sw_5_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v48:8 = fabric.instance @__chess_sw_8x8_3(%v47#1, %v49#0, %v39#3, %v57#2, %v115#3, %v116#2, %v123#1, %v124#0) {sym_name = "sw_5_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v49:8 = fabric.instance @__chess_sw_8x8_3(%v48#1, %v50#0, %v40#3, %v58#2, %v116#3, %v117#2, %v124#1, %v125#0) {sym_name = "sw_5_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v50:8 = fabric.instance @__chess_sw_8x8_3(%v49#1, %v51#0, %v41#3, %v59#2, %v117#3, %v118#2, %v125#1, %v126#0) {sym_name = "sw_5_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v51:8 = fabric.instance @__chess_sw_8x8_3(%v50#1, %v52#0, %v42#3, %v60#2, %v118#3, %v119#2, %v126#1, %v127#0) {sym_name = "sw_5_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v52:8 = fabric.instance @__chess_sw_8x8_3(%v51#1, %v53#0, %v43#3, %v61#2, %v119#3, %v120#2, %v127#1, %v128#0) {sym_name = "sw_5_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v53:5 = fabric.instance @__chess_sw_5x5_1(%v52#1, %v44#2, %v62#1, %v120#3, %v128#1) {sym_name = "sw_5_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v54:5 = fabric.instance @__chess_sw_5x5_1(%v55#0, %v45#2, %v63#1, %v121#2, %v129#0) {sym_name = "sw_6_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v55:8 = fabric.instance @__chess_sw_8x8_3(%v54#0, %v56#0, %v46#3, %v64#2, %v121#3, %v122#2, %v129#1, %v130#0) {sym_name = "sw_6_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v56:8 = fabric.instance @__chess_sw_8x8_3(%v55#1, %v57#0, %v47#3, %v65#2, %v122#3, %v123#2, %v130#1, %v131#0) {sym_name = "sw_6_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v57:8 = fabric.instance @__chess_sw_8x8_3(%v56#1, %v58#0, %v48#3, %v66#2, %v123#3, %v124#2, %v131#1, %v132#0) {sym_name = "sw_6_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v58:8 = fabric.instance @__chess_sw_8x8_3(%v57#1, %v59#0, %v49#3, %v67#2, %v124#3, %v125#2, %v132#1, %v133#0) {sym_name = "sw_6_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v59:8 = fabric.instance @__chess_sw_8x8_3(%v58#1, %v60#0, %v50#3, %v68#2, %v125#3, %v126#2, %v133#1, %v134#0) {sym_name = "sw_6_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v60:8 = fabric.instance @__chess_sw_8x8_3(%v59#1, %v61#0, %v51#3, %v69#2, %v126#3, %v127#2, %v134#1, %v135#0) {sym_name = "sw_6_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v61:8 = fabric.instance @__chess_sw_8x8_3(%v60#1, %v62#0, %v52#3, %v70#2, %v127#3, %v128#2, %v135#1, %v136#0) {sym_name = "sw_6_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v62:5 = fabric.instance @__chess_sw_5x5_1(%v61#1, %v53#2, %v71#1, %v128#3, %v136#1) {sym_name = "sw_6_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v63:5 = fabric.instance @__chess_sw_5x5_1(%v64#0, %v54#2, %v72#1, %v129#2, %v137#0) {sym_name = "sw_7_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v64:8 = fabric.instance @__chess_sw_8x8_3(%v63#0, %v65#0, %v55#3, %v73#2, %v129#3, %v130#2, %v137#1, %v138#0) {sym_name = "sw_7_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v65:8 = fabric.instance @__chess_sw_8x8_3(%v64#1, %v66#0, %v56#3, %v74#2, %v130#3, %v131#2, %v138#1, %v139#0) {sym_name = "sw_7_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v66:8 = fabric.instance @__chess_sw_8x8_3(%v65#1, %v67#0, %v57#3, %v75#2, %v131#3, %v132#2, %v139#1, %v140#0) {sym_name = "sw_7_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v67:8 = fabric.instance @__chess_sw_8x8_3(%v66#1, %v68#0, %v58#3, %v76#2, %v132#3, %v133#2, %v140#1, %v141#0) {sym_name = "sw_7_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v68:8 = fabric.instance @__chess_sw_8x8_3(%v67#1, %v69#0, %v59#3, %v77#2, %v133#3, %v134#2, %v141#1, %v142#0) {sym_name = "sw_7_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v69:8 = fabric.instance @__chess_sw_8x8_3(%v68#1, %v70#0, %v60#3, %v78#2, %v134#3, %v135#2, %v142#1, %v143#0) {sym_name = "sw_7_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v70:8 = fabric.instance @__chess_sw_8x8_3(%v69#1, %v71#0, %v61#3, %v79#2, %v135#3, %v136#2, %v143#1, %v144#0) {sym_name = "sw_7_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v71:5 = fabric.instance @__chess_sw_5x5_1(%v70#1, %v62#2, %v80#1, %v136#3, %v144#1) {sym_name = "sw_7_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v72:6 = fabric.instance @__chess_sw_3x6_4(%v73#0, %v63#2, %v137#2) {sym_name = "sw_8_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v73:5 = fabric.instance @__chess_sw_5x5_1(%v72#0, %v74#0, %v64#3, %v137#3, %v138#2) {sym_name = "sw_8_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v74:5 = fabric.instance @__chess_sw_5x5_1(%v73#1, %v75#0, %v65#3, %v138#3, %v139#2) {sym_name = "sw_8_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v75:5 = fabric.instance @__chess_sw_5x5_1(%v74#1, %v76#0, %v66#3, %v139#3, %v140#2) {sym_name = "sw_8_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v76:5 = fabric.instance @__chess_sw_5x5_1(%v75#1, %v77#0, %v67#3, %v140#3, %v141#2) {sym_name = "sw_8_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v77:5 = fabric.instance @__chess_sw_5x5_1(%v76#1, %v78#0, %v68#3, %v141#3, %v142#2) {sym_name = "sw_8_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v78:5 = fabric.instance @__chess_sw_5x5_1(%v77#1, %v79#0, %v69#3, %v142#3, %v143#2) {sym_name = "sw_8_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v79:5 = fabric.instance @__chess_sw_5x5_1(%v78#1, %v80#0, %v70#3, %v143#3, %v144#2) {sym_name = "sw_8_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v80:8 = fabric.instance @__chess_sw_3x8_5(%v79#1, %v71#2, %v144#3) {sym_name = "sw_8_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v81:4 = fabric.instance @SC-SPM_core_spe(%v0#2, %v1#3, %v9#3, %v10#4) {sym_name = "pe_0_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v82:4 = fabric.instance @SC-SPM_core_spe(%v1#4, %v2#3, %v10#5, %v11#4) {sym_name = "pe_0_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v83:4 = fabric.instance @SC-SPM_core_spe(%v2#4, %v3#3, %v11#5, %v12#4) {sym_name = "pe_0_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v84:4 = fabric.instance @SC-SPM_core_spe(%v3#4, %v4#3, %v12#5, %v13#4) {sym_name = "pe_0_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v85:4 = fabric.instance @SC-SPM_core_spe(%v4#4, %v5#3, %v13#5, %v14#4) {sym_name = "pe_0_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v86:4 = fabric.instance @SC-SPM_core_spe(%v5#4, %v6#3, %v14#5, %v15#4) {sym_name = "pe_0_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v87:4 = fabric.instance @SC-SPM_core_spe(%v6#4, %v7#3, %v15#5, %v16#4) {sym_name = "pe_0_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v88:4 = fabric.instance @SC-SPM_core_spe(%v7#4, %v8#2, %v16#5, %v17#3) {sym_name = "pe_0_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v89:4 = fabric.instance @SC-SPM_core_spe(%v9#4, %v10#6, %v18#3, %v19#4) {sym_name = "pe_1_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v90:4 = fabric.instance @SC-SPM_core_spe(%v10#7, %v11#6, %v19#5, %v20#4) {sym_name = "pe_1_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v91:4 = fabric.instance @SC-SPM_core_spe(%v11#7, %v12#6, %v20#5, %v21#4) {sym_name = "pe_1_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v92:4 = fabric.instance @SC-SPM_core_spe(%v12#7, %v13#6, %v21#5, %v22#4) {sym_name = "pe_1_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v93:4 = fabric.instance @SC-SPM_core_spe(%v13#7, %v14#6, %v22#5, %v23#4) {sym_name = "pe_1_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v94:4 = fabric.instance @SC-SPM_core_spe(%v14#7, %v15#6, %v23#5, %v24#4) {sym_name = "pe_1_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v95:4 = fabric.instance @SC-SPM_core_spe(%v15#7, %v16#6, %v24#5, %v25#4) {sym_name = "pe_1_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v96:4 = fabric.instance @SC-SPM_core_spe(%v16#7, %v17#4, %v25#5, %v26#3) {sym_name = "pe_1_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v97:4 = fabric.instance @SC-SPM_core_spe(%v18#4, %v19#6, %v27#3, %v28#4) {sym_name = "pe_2_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v98:4 = fabric.instance @SC-SPM_core_spe(%v19#7, %v20#6, %v28#5, %v29#4) {sym_name = "pe_2_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v99:4 = fabric.instance @SC-SPM_core_spe(%v20#7, %v21#6, %v29#5, %v30#4) {sym_name = "pe_2_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v100:4 = fabric.instance @SC-SPM_core_spe(%v21#7, %v22#6, %v30#5, %v31#4) {sym_name = "pe_2_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v101:4 = fabric.instance @SC-SPM_core_spe(%v22#7, %v23#6, %v31#5, %v32#4) {sym_name = "pe_2_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v102:4 = fabric.instance @SC-SPM_core_spe(%v23#7, %v24#6, %v32#5, %v33#4) {sym_name = "pe_2_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v103:4 = fabric.instance @SC-SPM_core_spe(%v24#7, %v25#6, %v33#5, %v34#4) {sym_name = "pe_2_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v104:4 = fabric.instance @SC-SPM_core_spe(%v25#7, %v26#4, %v34#5, %v35#3) {sym_name = "pe_2_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v105:4 = fabric.instance @SC-SPM_core_spe(%v27#4, %v28#6, %v36#3, %v37#4) {sym_name = "pe_3_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v106:4 = fabric.instance @SC-SPM_core_spe(%v28#7, %v29#6, %v37#5, %v38#4) {sym_name = "pe_3_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v107:4 = fabric.instance @SC-SPM_core_spe(%v29#7, %v30#6, %v38#5, %v39#4) {sym_name = "pe_3_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v108:4 = fabric.instance @SC-SPM_core_spe(%v30#7, %v31#6, %v39#5, %v40#4) {sym_name = "pe_3_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v109:4 = fabric.instance @SC-SPM_core_spe(%v31#7, %v32#6, %v40#5, %v41#4) {sym_name = "pe_3_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v110:4 = fabric.instance @SC-SPM_core_spe(%v32#7, %v33#6, %v41#5, %v42#4) {sym_name = "pe_3_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v111:4 = fabric.instance @SC-SPM_core_spe(%v33#7, %v34#6, %v42#5, %v43#4) {sym_name = "pe_3_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v112:4 = fabric.instance @SC-SPM_core_spe(%v34#7, %v35#4, %v43#5, %v44#3) {sym_name = "pe_3_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v113:4 = fabric.instance @SC-SPM_core_spe(%v36#4, %v37#6, %v45#3, %v46#4) {sym_name = "pe_4_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v114:4 = fabric.instance @SC-SPM_core_spe(%v37#7, %v38#6, %v46#5, %v47#4) {sym_name = "pe_4_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v115:4 = fabric.instance @SC-SPM_core_spe(%v38#7, %v39#6, %v47#5, %v48#4) {sym_name = "pe_4_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v116:4 = fabric.instance @SC-SPM_core_spe(%v39#7, %v40#6, %v48#5, %v49#4) {sym_name = "pe_4_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v117:4 = fabric.instance @SC-SPM_core_spe(%v40#7, %v41#6, %v49#5, %v50#4) {sym_name = "pe_4_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v118:4 = fabric.instance @SC-SPM_core_spe(%v41#7, %v42#6, %v50#5, %v51#4) {sym_name = "pe_4_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v119:4 = fabric.instance @SC-SPM_core_spe(%v42#7, %v43#6, %v51#5, %v52#4) {sym_name = "pe_4_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v120:4 = fabric.instance @SC-SPM_core_spe(%v43#7, %v44#4, %v52#5, %v53#3) {sym_name = "pe_4_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v121:4 = fabric.instance @SC-SPM_core_spe(%v45#4, %v46#6, %v54#3, %v55#4) {sym_name = "pe_5_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v122:4 = fabric.instance @SC-SPM_core_spe(%v46#7, %v47#6, %v55#5, %v56#4) {sym_name = "pe_5_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v123:4 = fabric.instance @SC-SPM_core_spe(%v47#7, %v48#6, %v56#5, %v57#4) {sym_name = "pe_5_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v124:4 = fabric.instance @SC-SPM_core_spe(%v48#7, %v49#6, %v57#5, %v58#4) {sym_name = "pe_5_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v125:4 = fabric.instance @SC-SPM_core_spe(%v49#7, %v50#6, %v58#5, %v59#4) {sym_name = "pe_5_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v126:4 = fabric.instance @SC-SPM_core_spe(%v50#7, %v51#6, %v59#5, %v60#4) {sym_name = "pe_5_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v127:4 = fabric.instance @SC-SPM_core_spe(%v51#7, %v52#6, %v60#5, %v61#4) {sym_name = "pe_5_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v128:4 = fabric.instance @SC-SPM_core_spe(%v52#7, %v53#4, %v61#5, %v62#3) {sym_name = "pe_5_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v129:4 = fabric.instance @SC-SPM_core_spe(%v54#4, %v55#6, %v63#3, %v64#4) {sym_name = "pe_6_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v130:4 = fabric.instance @SC-SPM_core_spe(%v55#7, %v56#6, %v64#5, %v65#4) {sym_name = "pe_6_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v131:4 = fabric.instance @SC-SPM_core_spe(%v56#7, %v57#6, %v65#5, %v66#4) {sym_name = "pe_6_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v132:4 = fabric.instance @SC-SPM_core_spe(%v57#7, %v58#6, %v66#5, %v67#4) {sym_name = "pe_6_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v133:4 = fabric.instance @SC-SPM_core_spe(%v58#7, %v59#6, %v67#5, %v68#4) {sym_name = "pe_6_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v134:4 = fabric.instance @SC-SPM_core_spe(%v59#7, %v60#6, %v68#5, %v69#4) {sym_name = "pe_6_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v135:4 = fabric.instance @SC-SPM_core_spe(%v60#7, %v61#6, %v69#5, %v70#4) {sym_name = "pe_6_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v136:4 = fabric.instance @SC-SPM_core_spe(%v61#7, %v62#4, %v70#5, %v71#3) {sym_name = "pe_6_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v137:4 = fabric.instance @SC-SPM_core_spe(%v63#4, %v64#6, %v72#2, %v73#3) {sym_name = "pe_7_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v138:4 = fabric.instance @SC-SPM_core_spe(%v64#7, %v65#6, %v73#4, %v74#3) {sym_name = "pe_7_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v139:4 = fabric.instance @SC-SPM_core_spe(%v65#7, %v66#6, %v74#4, %v75#3) {sym_name = "pe_7_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v140:4 = fabric.instance @SC-SPM_core_spe(%v66#7, %v67#6, %v75#4, %v76#3) {sym_name = "pe_7_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v141:4 = fabric.instance @SC-SPM_core_spe(%v67#7, %v68#6, %v76#4, %v77#3) {sym_name = "pe_7_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v142:4 = fabric.instance @SC-SPM_core_spe(%v68#7, %v69#6, %v77#4, %v78#3) {sym_name = "pe_7_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v143:4 = fabric.instance @SC-SPM_core_spe(%v69#7, %v70#6, %v78#4, %v79#3) {sym_name = "pe_7_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v144:4 = fabric.instance @SC-SPM_core_spe(%v70#7, %v71#4, %v79#4, %v80#2) {sym_name = "pe_7_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v145:3 = fabric.extmemory @extmem_0 [ldCount = 4, stCount = 2, lsqDepth = 0, memrefType = memref<?xi64>] (%mem0, %v72#3, %v72#4, %v72#5) : (memref<?xi64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v146:3 = fabric.extmemory @extmem_1 [ldCount = 4, stCount = 2, lsqDepth = 0, memrefType = memref<?xi64>] (%mem1, %v80#3, %v80#4, %v80#5) : (memref<?xi64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.yield %v80#6, %v80#7 : !fabric.bits<64>, !fabric.bits<64>
}
}
// CORE_TYPE_METADATA
// spm_capacity_bytes = 65536
