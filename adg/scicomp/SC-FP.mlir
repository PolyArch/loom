module {
fabric.module @SC-FP_core(%mem0: memref<?xi64>, %mem1: memref<?xi64>, %scalar0: !fabric.bits<64>, %scalar1: !fabric.bits<64>, %scalar2: !fabric.bits<64>, %scalar3: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) attributes {loom.decomposable = true, loom.extmem_ld_ports = 2, loom.extmem_st_ports = 1, loom.fp_add_units = 8, loom.fp_div_units = 1, loom.fp_mul_units = 6, loom.has_branch = false, loom.has_fma = true, loom.has_fp_min = true, loom.has_indirect_load = false, loom.has_rsqrt = true, loom.has_scatter_store = false, loom.int_alu_units = 4, loom.int_mul_units = 1, loom.routing_topology = "CHESS", loom.scicomp_khg_type = "SC-FP", loom.spm_ld_ports = 2, loom.spm_st_ports = 2, loom.sub_lane_bits = 32} {
  fabric.spatial_pe @SC-FP_core_spe(%in0: !fabric.bits<64>, %in1: !fabric.bits<64>, %in2: !fabric.bits<64>, %in3: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @SC-FP_core_const_int(%arg0: none) -> (i64) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0 : i64} : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_const_index(%arg0: none) -> (index) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0 : index} : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-FP_core_const_float(%arg0: none) -> (f64) [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0.000000e+00 : f64} : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_index_to_int(%arg0: index) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : index to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_int_to_index(%arg0: i64) -> (index) [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : i64 to index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-FP_core_stream(%arg0: index, %arg1: index, %arg2: index) -> (index, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.stream %arg0, %arg1, %arg2 {step_op = "+=", cont_cond = "<"} : (index, index, index) -> (index, i1)
      fabric.yield %0, %1 : index, i1
    }
    fabric.function_unit @SC-FP_core_mux_int(%arg0: index, %arg1: i64, %arg2: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_mux_none(%arg0: index, %arg1: none, %arg2: none) -> (none) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-FP_core_mux_index(%arg0: index, %arg1: index, %arg2: index) -> (index) [latency = 1, interval = 1] {
      %0 = handshake.mux %arg0 [%arg1, %arg2] : index, index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-FP_core_join(%arg0: none, %arg1: none, %arg2: none, %arg3: none) -> (none) [latency = 1, interval = 1] {
      %0 = handshake.join %arg0, %arg1, %arg2, %arg3 : none, none, none, none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-FP_core_gate_int(%arg0: i64, %arg1: i1) -> (i64, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : i64, i1 -> i64, i1
      fabric.yield %0, %1 : i64, i1
    }
    fabric.function_unit @SC-FP_core_gate_index(%arg0: index, %arg1: i1) -> (index, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : index, i1 -> index, i1
      fabric.yield %0, %1 : index, i1
    }
    fabric.function_unit @SC-FP_core_gate_float(%arg0: f64, %arg1: i1) -> (f64, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : f64, i1 -> f64, i1
      fabric.yield %0, %1 : f64, i1
    }
    fabric.function_unit @SC-FP_core_gate_i1(%arg0: i1, %arg1: i1) -> (i1, i1) [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %arg0, %arg1 : i1, i1 -> i1, i1
      fabric.yield %0, %1 : i1, i1
    }
    fabric.function_unit @SC-FP_core_carry_int(%arg0: i1, %arg1: i64, %arg2: i64) -> (i64) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, i64, i64 -> i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_carry_none(%arg0: i1, %arg1: none, %arg2: none) -> (none) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, none, none -> none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-FP_core_carry_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = -1, interval = -1] {
      %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, f64, f64 -> f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_cond_br_int(%arg0: i1, %arg1: i64) -> (i64, i64) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : i64
      fabric.yield %0, %1 : i64, i64
    }
    fabric.function_unit @SC-FP_core_cond_br_none(%arg0: i1, %arg1: none) -> (none, none) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : none
      fabric.yield %0, %1 : none, none
    }
    fabric.function_unit @SC-FP_core_cond_br_float(%arg0: i1, %arg1: f64) -> (f64, f64) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %arg0, %arg1 : f64
      fabric.yield %0, %1 : f64, f64
    }
    fabric.function_unit @SC-FP_core_invariant_int(%arg0: i1, %arg1: i64) -> (i64) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, i64 -> i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_invariant_index(%arg0: i1, %arg1: index) -> (index) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, index -> index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-FP_core_invariant_float(%arg0: i1, %arg1: f64) -> (f64) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, f64 -> f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_invariant_none(%arg0: i1, %arg1: none) -> (none) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, none -> none
      fabric.yield %0 : none
    }
    fabric.function_unit @SC-FP_core_invariant_i1(%arg0: i1, %arg1: i1) -> (i1) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %arg0, %arg1 : i1, i1 -> i1
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_load(%arg0: index, %arg1: i64, %arg2: none) -> (i64, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i64
      fabric.yield %0, %1 : i64, index
    }
    fabric.function_unit @SC-FP_core_store(%arg0: index, %arg1: i64, %arg2: none) -> (i64, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.store [%arg0] %arg1, %arg2 : index, i64
      fabric.yield %0, %1 : i64, index
    }
    fabric.function_unit @SC-FP_core_select_int(%arg0: i1, %arg1: i64, %arg2: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_select_index(%arg0: i1, %arg1: index, %arg2: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-FP_core_cmpi_int(%arg0: i64, %arg1: i64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpi slt, %arg0, %arg1 : i64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_addi_index(%arg0: index, %arg1: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-FP_core_muli_index(%arg0: index, %arg1: index) -> (index) [latency = 1, interval = 1] {
      %0 = arith.muli %arg0, %arg1 : index
      fabric.yield %0 : index
    }
    fabric.function_unit @SC-FP_core_alu0_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu0_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu0_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu0_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu0_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu0_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu0_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu0_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu1_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu2_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_addi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_subi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.subi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_andi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.andi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_ori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.ori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_xori(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.xori %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_shli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_shrsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_alu3_shrui(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.shrui %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_mul0_muli(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.muli %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_mul0_divsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.divsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_mul0_remsi(%arg0: i64, %arg1: i64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.remsi %arg0, %arg1 : i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp0_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp0_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp0_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp0_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp0_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp0_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp0_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp0_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp0_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp1_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp1_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp1_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp1_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp1_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp1_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp1_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp1_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp1_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp2_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp2_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp2_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp2_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp2_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp2_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp2_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp2_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp2_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp3_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp3_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp3_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp3_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp3_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp3_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp3_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp3_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp3_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp4_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp4_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp4_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp4_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp4_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp4_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp4_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp4_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp4_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp5_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp5_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp5_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp5_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp5_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp5_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp5_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp5_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp5_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp6_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp6_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp6_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp6_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp6_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp6_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp6_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp6_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp6_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp7_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp7_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp7_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp7_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp7_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp7_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp7_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp7_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp7_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp8_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp8_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp8_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp8_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp8_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp8_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp8_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp8_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp8_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp9_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp9_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp9_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp9_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp9_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp9_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp9_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp9_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp9_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp10_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp10_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp10_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp10_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp10_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp10_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp10_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp10_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp10_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp11_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp11_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp11_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp11_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp11_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp11_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp11_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp11_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp11_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp12_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp12_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp12_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp12_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp12_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp12_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp12_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp12_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp12_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp13_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp13_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp13_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp13_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp13_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp13_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp13_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp13_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp13_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp14_addf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.addf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp14_subf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.subf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp14_mulf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.mulf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp14_divf(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.divf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp14_cmpf(%arg0: f64, %arg1: f64) -> (i1) [latency = 1, interval = 1] {
      %0 = arith.cmpf olt, %arg0, %arg1 : f64
      fabric.yield %0 : i1
    }
    fabric.function_unit @SC-FP_core_fp14_select_float(%arg0: i1, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.select %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp14_sitofp(%arg0: i64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.sitofp %arg0 : i64 to f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp14_fptosi(%arg0: f64) -> (i64) [latency = 1, interval = 1] {
      %0 = arith.fptosi %arg0 : f64 to i64
      fabric.yield %0 : i64
    }
    fabric.function_unit @SC-FP_core_fp14_negf(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.negf %arg0 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp_fma(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = math.fma %arg0, %arg1, %arg2 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp_rsqrt(%arg0: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = math.rsqrt %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.function_unit @SC-FP_core_fp_min(%arg0: f64, %arg1: f64) -> (f64) [latency = 1, interval = 1] {
      %0 = arith.minimumf %arg0, %arg1 : f64
      fabric.yield %0 : f64
    }
    fabric.yield
  }
  fabric.spatial_sw @__chess_sw_10x3_0 [connectivity_table = ["1111111111", "1111111111", "1111111111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_5x5_1 [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_6x3_2 [connectivity_table = ["111111", "111111", "111111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_8x8_3 [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_3x6_4 [connectivity_table = ["111", "111", "111", "111", "111", "111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_sw @__chess_sw_3x8_5 [connectivity_table = ["111", "111", "111", "111", "111", "111", "111", "111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v0:3 = fabric.instance @__chess_sw_10x3_0(%v1#0, %v13#1, %v169#0, %v313#0, %v313#1, %v313#2, %scalar0, %scalar1, %scalar2, %scalar3) {sym_name = "sw_0_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v1:5 = fabric.instance @__chess_sw_5x5_1(%v0#0, %v2#0, %v14#2, %v169#1, %v170#0) {sym_name = "sw_0_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v2:5 = fabric.instance @__chess_sw_5x5_1(%v1#1, %v3#0, %v15#2, %v170#1, %v171#0) {sym_name = "sw_0_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v3:5 = fabric.instance @__chess_sw_5x5_1(%v2#1, %v4#0, %v16#2, %v171#1, %v172#0) {sym_name = "sw_0_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v4:5 = fabric.instance @__chess_sw_5x5_1(%v3#1, %v5#0, %v17#2, %v172#1, %v173#0) {sym_name = "sw_0_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v5:5 = fabric.instance @__chess_sw_5x5_1(%v4#1, %v6#0, %v18#2, %v173#1, %v174#0) {sym_name = "sw_0_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v6:5 = fabric.instance @__chess_sw_5x5_1(%v5#1, %v7#0, %v19#2, %v174#1, %v175#0) {sym_name = "sw_0_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v7:5 = fabric.instance @__chess_sw_5x5_1(%v6#1, %v8#0, %v20#2, %v175#1, %v176#0) {sym_name = "sw_0_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v8:5 = fabric.instance @__chess_sw_5x5_1(%v7#1, %v9#0, %v21#2, %v176#1, %v177#0) {sym_name = "sw_0_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v9:5 = fabric.instance @__chess_sw_5x5_1(%v8#1, %v10#0, %v22#2, %v177#1, %v178#0) {sym_name = "sw_0_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v10:5 = fabric.instance @__chess_sw_5x5_1(%v9#1, %v11#0, %v23#2, %v178#1, %v179#0) {sym_name = "sw_0_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v11:5 = fabric.instance @__chess_sw_5x5_1(%v10#1, %v12#0, %v24#2, %v179#1, %v180#0) {sym_name = "sw_0_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v12:3 = fabric.instance @__chess_sw_6x3_2(%v11#1, %v25#1, %v180#1, %v314#0, %v314#1, %v314#2) {sym_name = "sw_0_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v13:5 = fabric.instance @__chess_sw_5x5_1(%v14#0, %v0#1, %v26#1, %v169#2, %v181#0) {sym_name = "sw_1_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v14:8 = fabric.instance @__chess_sw_8x8_3(%v13#0, %v15#0, %v1#2, %v27#2, %v169#3, %v170#2, %v181#1, %v182#0) {sym_name = "sw_1_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v15:8 = fabric.instance @__chess_sw_8x8_3(%v14#1, %v16#0, %v2#2, %v28#2, %v170#3, %v171#2, %v182#1, %v183#0) {sym_name = "sw_1_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v16:8 = fabric.instance @__chess_sw_8x8_3(%v15#1, %v17#0, %v3#2, %v29#2, %v171#3, %v172#2, %v183#1, %v184#0) {sym_name = "sw_1_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v17:8 = fabric.instance @__chess_sw_8x8_3(%v16#1, %v18#0, %v4#2, %v30#2, %v172#3, %v173#2, %v184#1, %v185#0) {sym_name = "sw_1_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v18:8 = fabric.instance @__chess_sw_8x8_3(%v17#1, %v19#0, %v5#2, %v31#2, %v173#3, %v174#2, %v185#1, %v186#0) {sym_name = "sw_1_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v19:8 = fabric.instance @__chess_sw_8x8_3(%v18#1, %v20#0, %v6#2, %v32#2, %v174#3, %v175#2, %v186#1, %v187#0) {sym_name = "sw_1_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v20:8 = fabric.instance @__chess_sw_8x8_3(%v19#1, %v21#0, %v7#2, %v33#2, %v175#3, %v176#2, %v187#1, %v188#0) {sym_name = "sw_1_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v21:8 = fabric.instance @__chess_sw_8x8_3(%v20#1, %v22#0, %v8#2, %v34#2, %v176#3, %v177#2, %v188#1, %v189#0) {sym_name = "sw_1_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v22:8 = fabric.instance @__chess_sw_8x8_3(%v21#1, %v23#0, %v9#2, %v35#2, %v177#3, %v178#2, %v189#1, %v190#0) {sym_name = "sw_1_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v23:8 = fabric.instance @__chess_sw_8x8_3(%v22#1, %v24#0, %v10#2, %v36#2, %v178#3, %v179#2, %v190#1, %v191#0) {sym_name = "sw_1_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v24:8 = fabric.instance @__chess_sw_8x8_3(%v23#1, %v25#0, %v11#2, %v37#2, %v179#3, %v180#2, %v191#1, %v192#0) {sym_name = "sw_1_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v25:5 = fabric.instance @__chess_sw_5x5_1(%v24#1, %v12#1, %v38#1, %v180#3, %v192#1) {sym_name = "sw_1_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v26:5 = fabric.instance @__chess_sw_5x5_1(%v27#0, %v13#2, %v39#1, %v181#2, %v193#0) {sym_name = "sw_2_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v27:8 = fabric.instance @__chess_sw_8x8_3(%v26#0, %v28#0, %v14#3, %v40#2, %v181#3, %v182#2, %v193#1, %v194#0) {sym_name = "sw_2_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v28:8 = fabric.instance @__chess_sw_8x8_3(%v27#1, %v29#0, %v15#3, %v41#2, %v182#3, %v183#2, %v194#1, %v195#0) {sym_name = "sw_2_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v29:8 = fabric.instance @__chess_sw_8x8_3(%v28#1, %v30#0, %v16#3, %v42#2, %v183#3, %v184#2, %v195#1, %v196#0) {sym_name = "sw_2_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v30:8 = fabric.instance @__chess_sw_8x8_3(%v29#1, %v31#0, %v17#3, %v43#2, %v184#3, %v185#2, %v196#1, %v197#0) {sym_name = "sw_2_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v31:8 = fabric.instance @__chess_sw_8x8_3(%v30#1, %v32#0, %v18#3, %v44#2, %v185#3, %v186#2, %v197#1, %v198#0) {sym_name = "sw_2_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v32:8 = fabric.instance @__chess_sw_8x8_3(%v31#1, %v33#0, %v19#3, %v45#2, %v186#3, %v187#2, %v198#1, %v199#0) {sym_name = "sw_2_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v33:8 = fabric.instance @__chess_sw_8x8_3(%v32#1, %v34#0, %v20#3, %v46#2, %v187#3, %v188#2, %v199#1, %v200#0) {sym_name = "sw_2_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v34:8 = fabric.instance @__chess_sw_8x8_3(%v33#1, %v35#0, %v21#3, %v47#2, %v188#3, %v189#2, %v200#1, %v201#0) {sym_name = "sw_2_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v35:8 = fabric.instance @__chess_sw_8x8_3(%v34#1, %v36#0, %v22#3, %v48#2, %v189#3, %v190#2, %v201#1, %v202#0) {sym_name = "sw_2_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v36:8 = fabric.instance @__chess_sw_8x8_3(%v35#1, %v37#0, %v23#3, %v49#2, %v190#3, %v191#2, %v202#1, %v203#0) {sym_name = "sw_2_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v37:8 = fabric.instance @__chess_sw_8x8_3(%v36#1, %v38#0, %v24#3, %v50#2, %v191#3, %v192#2, %v203#1, %v204#0) {sym_name = "sw_2_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v38:5 = fabric.instance @__chess_sw_5x5_1(%v37#1, %v25#2, %v51#1, %v192#3, %v204#1) {sym_name = "sw_2_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v39:5 = fabric.instance @__chess_sw_5x5_1(%v40#0, %v26#2, %v52#1, %v193#2, %v205#0) {sym_name = "sw_3_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v40:8 = fabric.instance @__chess_sw_8x8_3(%v39#0, %v41#0, %v27#3, %v53#2, %v193#3, %v194#2, %v205#1, %v206#0) {sym_name = "sw_3_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v41:8 = fabric.instance @__chess_sw_8x8_3(%v40#1, %v42#0, %v28#3, %v54#2, %v194#3, %v195#2, %v206#1, %v207#0) {sym_name = "sw_3_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v42:8 = fabric.instance @__chess_sw_8x8_3(%v41#1, %v43#0, %v29#3, %v55#2, %v195#3, %v196#2, %v207#1, %v208#0) {sym_name = "sw_3_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v43:8 = fabric.instance @__chess_sw_8x8_3(%v42#1, %v44#0, %v30#3, %v56#2, %v196#3, %v197#2, %v208#1, %v209#0) {sym_name = "sw_3_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v44:8 = fabric.instance @__chess_sw_8x8_3(%v43#1, %v45#0, %v31#3, %v57#2, %v197#3, %v198#2, %v209#1, %v210#0) {sym_name = "sw_3_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v45:8 = fabric.instance @__chess_sw_8x8_3(%v44#1, %v46#0, %v32#3, %v58#2, %v198#3, %v199#2, %v210#1, %v211#0) {sym_name = "sw_3_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v46:8 = fabric.instance @__chess_sw_8x8_3(%v45#1, %v47#0, %v33#3, %v59#2, %v199#3, %v200#2, %v211#1, %v212#0) {sym_name = "sw_3_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v47:8 = fabric.instance @__chess_sw_8x8_3(%v46#1, %v48#0, %v34#3, %v60#2, %v200#3, %v201#2, %v212#1, %v213#0) {sym_name = "sw_3_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v48:8 = fabric.instance @__chess_sw_8x8_3(%v47#1, %v49#0, %v35#3, %v61#2, %v201#3, %v202#2, %v213#1, %v214#0) {sym_name = "sw_3_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v49:8 = fabric.instance @__chess_sw_8x8_3(%v48#1, %v50#0, %v36#3, %v62#2, %v202#3, %v203#2, %v214#1, %v215#0) {sym_name = "sw_3_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v50:8 = fabric.instance @__chess_sw_8x8_3(%v49#1, %v51#0, %v37#3, %v63#2, %v203#3, %v204#2, %v215#1, %v216#0) {sym_name = "sw_3_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v51:5 = fabric.instance @__chess_sw_5x5_1(%v50#1, %v38#2, %v64#1, %v204#3, %v216#1) {sym_name = "sw_3_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v52:5 = fabric.instance @__chess_sw_5x5_1(%v53#0, %v39#2, %v65#1, %v205#2, %v217#0) {sym_name = "sw_4_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v53:8 = fabric.instance @__chess_sw_8x8_3(%v52#0, %v54#0, %v40#3, %v66#2, %v205#3, %v206#2, %v217#1, %v218#0) {sym_name = "sw_4_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v54:8 = fabric.instance @__chess_sw_8x8_3(%v53#1, %v55#0, %v41#3, %v67#2, %v206#3, %v207#2, %v218#1, %v219#0) {sym_name = "sw_4_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v55:8 = fabric.instance @__chess_sw_8x8_3(%v54#1, %v56#0, %v42#3, %v68#2, %v207#3, %v208#2, %v219#1, %v220#0) {sym_name = "sw_4_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v56:8 = fabric.instance @__chess_sw_8x8_3(%v55#1, %v57#0, %v43#3, %v69#2, %v208#3, %v209#2, %v220#1, %v221#0) {sym_name = "sw_4_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v57:8 = fabric.instance @__chess_sw_8x8_3(%v56#1, %v58#0, %v44#3, %v70#2, %v209#3, %v210#2, %v221#1, %v222#0) {sym_name = "sw_4_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v58:8 = fabric.instance @__chess_sw_8x8_3(%v57#1, %v59#0, %v45#3, %v71#2, %v210#3, %v211#2, %v222#1, %v223#0) {sym_name = "sw_4_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v59:8 = fabric.instance @__chess_sw_8x8_3(%v58#1, %v60#0, %v46#3, %v72#2, %v211#3, %v212#2, %v223#1, %v224#0) {sym_name = "sw_4_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v60:8 = fabric.instance @__chess_sw_8x8_3(%v59#1, %v61#0, %v47#3, %v73#2, %v212#3, %v213#2, %v224#1, %v225#0) {sym_name = "sw_4_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v61:8 = fabric.instance @__chess_sw_8x8_3(%v60#1, %v62#0, %v48#3, %v74#2, %v213#3, %v214#2, %v225#1, %v226#0) {sym_name = "sw_4_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v62:8 = fabric.instance @__chess_sw_8x8_3(%v61#1, %v63#0, %v49#3, %v75#2, %v214#3, %v215#2, %v226#1, %v227#0) {sym_name = "sw_4_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v63:8 = fabric.instance @__chess_sw_8x8_3(%v62#1, %v64#0, %v50#3, %v76#2, %v215#3, %v216#2, %v227#1, %v228#0) {sym_name = "sw_4_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v64:5 = fabric.instance @__chess_sw_5x5_1(%v63#1, %v51#2, %v77#1, %v216#3, %v228#1) {sym_name = "sw_4_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v65:5 = fabric.instance @__chess_sw_5x5_1(%v66#0, %v52#2, %v78#1, %v217#2, %v229#0) {sym_name = "sw_5_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v66:8 = fabric.instance @__chess_sw_8x8_3(%v65#0, %v67#0, %v53#3, %v79#2, %v217#3, %v218#2, %v229#1, %v230#0) {sym_name = "sw_5_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v67:8 = fabric.instance @__chess_sw_8x8_3(%v66#1, %v68#0, %v54#3, %v80#2, %v218#3, %v219#2, %v230#1, %v231#0) {sym_name = "sw_5_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v68:8 = fabric.instance @__chess_sw_8x8_3(%v67#1, %v69#0, %v55#3, %v81#2, %v219#3, %v220#2, %v231#1, %v232#0) {sym_name = "sw_5_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v69:8 = fabric.instance @__chess_sw_8x8_3(%v68#1, %v70#0, %v56#3, %v82#2, %v220#3, %v221#2, %v232#1, %v233#0) {sym_name = "sw_5_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v70:8 = fabric.instance @__chess_sw_8x8_3(%v69#1, %v71#0, %v57#3, %v83#2, %v221#3, %v222#2, %v233#1, %v234#0) {sym_name = "sw_5_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v71:8 = fabric.instance @__chess_sw_8x8_3(%v70#1, %v72#0, %v58#3, %v84#2, %v222#3, %v223#2, %v234#1, %v235#0) {sym_name = "sw_5_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v72:8 = fabric.instance @__chess_sw_8x8_3(%v71#1, %v73#0, %v59#3, %v85#2, %v223#3, %v224#2, %v235#1, %v236#0) {sym_name = "sw_5_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v73:8 = fabric.instance @__chess_sw_8x8_3(%v72#1, %v74#0, %v60#3, %v86#2, %v224#3, %v225#2, %v236#1, %v237#0) {sym_name = "sw_5_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v74:8 = fabric.instance @__chess_sw_8x8_3(%v73#1, %v75#0, %v61#3, %v87#2, %v225#3, %v226#2, %v237#1, %v238#0) {sym_name = "sw_5_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v75:8 = fabric.instance @__chess_sw_8x8_3(%v74#1, %v76#0, %v62#3, %v88#2, %v226#3, %v227#2, %v238#1, %v239#0) {sym_name = "sw_5_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v76:8 = fabric.instance @__chess_sw_8x8_3(%v75#1, %v77#0, %v63#3, %v89#2, %v227#3, %v228#2, %v239#1, %v240#0) {sym_name = "sw_5_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v77:5 = fabric.instance @__chess_sw_5x5_1(%v76#1, %v64#2, %v90#1, %v228#3, %v240#1) {sym_name = "sw_5_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v78:5 = fabric.instance @__chess_sw_5x5_1(%v79#0, %v65#2, %v91#1, %v229#2, %v241#0) {sym_name = "sw_6_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v79:8 = fabric.instance @__chess_sw_8x8_3(%v78#0, %v80#0, %v66#3, %v92#2, %v229#3, %v230#2, %v241#1, %v242#0) {sym_name = "sw_6_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v80:8 = fabric.instance @__chess_sw_8x8_3(%v79#1, %v81#0, %v67#3, %v93#2, %v230#3, %v231#2, %v242#1, %v243#0) {sym_name = "sw_6_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v81:8 = fabric.instance @__chess_sw_8x8_3(%v80#1, %v82#0, %v68#3, %v94#2, %v231#3, %v232#2, %v243#1, %v244#0) {sym_name = "sw_6_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v82:8 = fabric.instance @__chess_sw_8x8_3(%v81#1, %v83#0, %v69#3, %v95#2, %v232#3, %v233#2, %v244#1, %v245#0) {sym_name = "sw_6_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v83:8 = fabric.instance @__chess_sw_8x8_3(%v82#1, %v84#0, %v70#3, %v96#2, %v233#3, %v234#2, %v245#1, %v246#0) {sym_name = "sw_6_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v84:8 = fabric.instance @__chess_sw_8x8_3(%v83#1, %v85#0, %v71#3, %v97#2, %v234#3, %v235#2, %v246#1, %v247#0) {sym_name = "sw_6_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v85:8 = fabric.instance @__chess_sw_8x8_3(%v84#1, %v86#0, %v72#3, %v98#2, %v235#3, %v236#2, %v247#1, %v248#0) {sym_name = "sw_6_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v86:8 = fabric.instance @__chess_sw_8x8_3(%v85#1, %v87#0, %v73#3, %v99#2, %v236#3, %v237#2, %v248#1, %v249#0) {sym_name = "sw_6_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v87:8 = fabric.instance @__chess_sw_8x8_3(%v86#1, %v88#0, %v74#3, %v100#2, %v237#3, %v238#2, %v249#1, %v250#0) {sym_name = "sw_6_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v88:8 = fabric.instance @__chess_sw_8x8_3(%v87#1, %v89#0, %v75#3, %v101#2, %v238#3, %v239#2, %v250#1, %v251#0) {sym_name = "sw_6_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v89:8 = fabric.instance @__chess_sw_8x8_3(%v88#1, %v90#0, %v76#3, %v102#2, %v239#3, %v240#2, %v251#1, %v252#0) {sym_name = "sw_6_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v90:5 = fabric.instance @__chess_sw_5x5_1(%v89#1, %v77#2, %v103#1, %v240#3, %v252#1) {sym_name = "sw_6_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v91:5 = fabric.instance @__chess_sw_5x5_1(%v92#0, %v78#2, %v104#1, %v241#2, %v253#0) {sym_name = "sw_7_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v92:8 = fabric.instance @__chess_sw_8x8_3(%v91#0, %v93#0, %v79#3, %v105#2, %v241#3, %v242#2, %v253#1, %v254#0) {sym_name = "sw_7_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v93:8 = fabric.instance @__chess_sw_8x8_3(%v92#1, %v94#0, %v80#3, %v106#2, %v242#3, %v243#2, %v254#1, %v255#0) {sym_name = "sw_7_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v94:8 = fabric.instance @__chess_sw_8x8_3(%v93#1, %v95#0, %v81#3, %v107#2, %v243#3, %v244#2, %v255#1, %v256#0) {sym_name = "sw_7_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v95:8 = fabric.instance @__chess_sw_8x8_3(%v94#1, %v96#0, %v82#3, %v108#2, %v244#3, %v245#2, %v256#1, %v257#0) {sym_name = "sw_7_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v96:8 = fabric.instance @__chess_sw_8x8_3(%v95#1, %v97#0, %v83#3, %v109#2, %v245#3, %v246#2, %v257#1, %v258#0) {sym_name = "sw_7_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v97:8 = fabric.instance @__chess_sw_8x8_3(%v96#1, %v98#0, %v84#3, %v110#2, %v246#3, %v247#2, %v258#1, %v259#0) {sym_name = "sw_7_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v98:8 = fabric.instance @__chess_sw_8x8_3(%v97#1, %v99#0, %v85#3, %v111#2, %v247#3, %v248#2, %v259#1, %v260#0) {sym_name = "sw_7_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v99:8 = fabric.instance @__chess_sw_8x8_3(%v98#1, %v100#0, %v86#3, %v112#2, %v248#3, %v249#2, %v260#1, %v261#0) {sym_name = "sw_7_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v100:8 = fabric.instance @__chess_sw_8x8_3(%v99#1, %v101#0, %v87#3, %v113#2, %v249#3, %v250#2, %v261#1, %v262#0) {sym_name = "sw_7_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v101:8 = fabric.instance @__chess_sw_8x8_3(%v100#1, %v102#0, %v88#3, %v114#2, %v250#3, %v251#2, %v262#1, %v263#0) {sym_name = "sw_7_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v102:8 = fabric.instance @__chess_sw_8x8_3(%v101#1, %v103#0, %v89#3, %v115#2, %v251#3, %v252#2, %v263#1, %v264#0) {sym_name = "sw_7_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v103:5 = fabric.instance @__chess_sw_5x5_1(%v102#1, %v90#2, %v116#1, %v252#3, %v264#1) {sym_name = "sw_7_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v104:5 = fabric.instance @__chess_sw_5x5_1(%v105#0, %v91#2, %v117#1, %v253#2, %v265#0) {sym_name = "sw_8_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v105:8 = fabric.instance @__chess_sw_8x8_3(%v104#0, %v106#0, %v92#3, %v118#2, %v253#3, %v254#2, %v265#1, %v266#0) {sym_name = "sw_8_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v106:8 = fabric.instance @__chess_sw_8x8_3(%v105#1, %v107#0, %v93#3, %v119#2, %v254#3, %v255#2, %v266#1, %v267#0) {sym_name = "sw_8_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v107:8 = fabric.instance @__chess_sw_8x8_3(%v106#1, %v108#0, %v94#3, %v120#2, %v255#3, %v256#2, %v267#1, %v268#0) {sym_name = "sw_8_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v108:8 = fabric.instance @__chess_sw_8x8_3(%v107#1, %v109#0, %v95#3, %v121#2, %v256#3, %v257#2, %v268#1, %v269#0) {sym_name = "sw_8_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v109:8 = fabric.instance @__chess_sw_8x8_3(%v108#1, %v110#0, %v96#3, %v122#2, %v257#3, %v258#2, %v269#1, %v270#0) {sym_name = "sw_8_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v110:8 = fabric.instance @__chess_sw_8x8_3(%v109#1, %v111#0, %v97#3, %v123#2, %v258#3, %v259#2, %v270#1, %v271#0) {sym_name = "sw_8_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v111:8 = fabric.instance @__chess_sw_8x8_3(%v110#1, %v112#0, %v98#3, %v124#2, %v259#3, %v260#2, %v271#1, %v272#0) {sym_name = "sw_8_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v112:8 = fabric.instance @__chess_sw_8x8_3(%v111#1, %v113#0, %v99#3, %v125#2, %v260#3, %v261#2, %v272#1, %v273#0) {sym_name = "sw_8_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v113:8 = fabric.instance @__chess_sw_8x8_3(%v112#1, %v114#0, %v100#3, %v126#2, %v261#3, %v262#2, %v273#1, %v274#0) {sym_name = "sw_8_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v114:8 = fabric.instance @__chess_sw_8x8_3(%v113#1, %v115#0, %v101#3, %v127#2, %v262#3, %v263#2, %v274#1, %v275#0) {sym_name = "sw_8_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v115:8 = fabric.instance @__chess_sw_8x8_3(%v114#1, %v116#0, %v102#3, %v128#2, %v263#3, %v264#2, %v275#1, %v276#0) {sym_name = "sw_8_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v116:5 = fabric.instance @__chess_sw_5x5_1(%v115#1, %v103#2, %v129#1, %v264#3, %v276#1) {sym_name = "sw_8_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v117:5 = fabric.instance @__chess_sw_5x5_1(%v118#0, %v104#2, %v130#1, %v265#2, %v277#0) {sym_name = "sw_9_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v118:8 = fabric.instance @__chess_sw_8x8_3(%v117#0, %v119#0, %v105#3, %v131#2, %v265#3, %v266#2, %v277#1, %v278#0) {sym_name = "sw_9_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v119:8 = fabric.instance @__chess_sw_8x8_3(%v118#1, %v120#0, %v106#3, %v132#2, %v266#3, %v267#2, %v278#1, %v279#0) {sym_name = "sw_9_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v120:8 = fabric.instance @__chess_sw_8x8_3(%v119#1, %v121#0, %v107#3, %v133#2, %v267#3, %v268#2, %v279#1, %v280#0) {sym_name = "sw_9_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v121:8 = fabric.instance @__chess_sw_8x8_3(%v120#1, %v122#0, %v108#3, %v134#2, %v268#3, %v269#2, %v280#1, %v281#0) {sym_name = "sw_9_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v122:8 = fabric.instance @__chess_sw_8x8_3(%v121#1, %v123#0, %v109#3, %v135#2, %v269#3, %v270#2, %v281#1, %v282#0) {sym_name = "sw_9_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v123:8 = fabric.instance @__chess_sw_8x8_3(%v122#1, %v124#0, %v110#3, %v136#2, %v270#3, %v271#2, %v282#1, %v283#0) {sym_name = "sw_9_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v124:8 = fabric.instance @__chess_sw_8x8_3(%v123#1, %v125#0, %v111#3, %v137#2, %v271#3, %v272#2, %v283#1, %v284#0) {sym_name = "sw_9_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v125:8 = fabric.instance @__chess_sw_8x8_3(%v124#1, %v126#0, %v112#3, %v138#2, %v272#3, %v273#2, %v284#1, %v285#0) {sym_name = "sw_9_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v126:8 = fabric.instance @__chess_sw_8x8_3(%v125#1, %v127#0, %v113#3, %v139#2, %v273#3, %v274#2, %v285#1, %v286#0) {sym_name = "sw_9_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v127:8 = fabric.instance @__chess_sw_8x8_3(%v126#1, %v128#0, %v114#3, %v140#2, %v274#3, %v275#2, %v286#1, %v287#0) {sym_name = "sw_9_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v128:8 = fabric.instance @__chess_sw_8x8_3(%v127#1, %v129#0, %v115#3, %v141#2, %v275#3, %v276#2, %v287#1, %v288#0) {sym_name = "sw_9_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v129:5 = fabric.instance @__chess_sw_5x5_1(%v128#1, %v116#2, %v142#1, %v276#3, %v288#1) {sym_name = "sw_9_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v130:5 = fabric.instance @__chess_sw_5x5_1(%v131#0, %v117#2, %v143#1, %v277#2, %v289#0) {sym_name = "sw_10_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v131:8 = fabric.instance @__chess_sw_8x8_3(%v130#0, %v132#0, %v118#3, %v144#2, %v277#3, %v278#2, %v289#1, %v290#0) {sym_name = "sw_10_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v132:8 = fabric.instance @__chess_sw_8x8_3(%v131#1, %v133#0, %v119#3, %v145#2, %v278#3, %v279#2, %v290#1, %v291#0) {sym_name = "sw_10_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v133:8 = fabric.instance @__chess_sw_8x8_3(%v132#1, %v134#0, %v120#3, %v146#2, %v279#3, %v280#2, %v291#1, %v292#0) {sym_name = "sw_10_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v134:8 = fabric.instance @__chess_sw_8x8_3(%v133#1, %v135#0, %v121#3, %v147#2, %v280#3, %v281#2, %v292#1, %v293#0) {sym_name = "sw_10_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v135:8 = fabric.instance @__chess_sw_8x8_3(%v134#1, %v136#0, %v122#3, %v148#2, %v281#3, %v282#2, %v293#1, %v294#0) {sym_name = "sw_10_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v136:8 = fabric.instance @__chess_sw_8x8_3(%v135#1, %v137#0, %v123#3, %v149#2, %v282#3, %v283#2, %v294#1, %v295#0) {sym_name = "sw_10_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v137:8 = fabric.instance @__chess_sw_8x8_3(%v136#1, %v138#0, %v124#3, %v150#2, %v283#3, %v284#2, %v295#1, %v296#0) {sym_name = "sw_10_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v138:8 = fabric.instance @__chess_sw_8x8_3(%v137#1, %v139#0, %v125#3, %v151#2, %v284#3, %v285#2, %v296#1, %v297#0) {sym_name = "sw_10_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v139:8 = fabric.instance @__chess_sw_8x8_3(%v138#1, %v140#0, %v126#3, %v152#2, %v285#3, %v286#2, %v297#1, %v298#0) {sym_name = "sw_10_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v140:8 = fabric.instance @__chess_sw_8x8_3(%v139#1, %v141#0, %v127#3, %v153#2, %v286#3, %v287#2, %v298#1, %v299#0) {sym_name = "sw_10_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v141:8 = fabric.instance @__chess_sw_8x8_3(%v140#1, %v142#0, %v128#3, %v154#2, %v287#3, %v288#2, %v299#1, %v300#0) {sym_name = "sw_10_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v142:5 = fabric.instance @__chess_sw_5x5_1(%v141#1, %v129#2, %v155#1, %v288#3, %v300#1) {sym_name = "sw_10_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v143:5 = fabric.instance @__chess_sw_5x5_1(%v144#0, %v130#2, %v156#1, %v289#2, %v301#0) {sym_name = "sw_11_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v144:8 = fabric.instance @__chess_sw_8x8_3(%v143#0, %v145#0, %v131#3, %v157#2, %v289#3, %v290#2, %v301#1, %v302#0) {sym_name = "sw_11_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v145:8 = fabric.instance @__chess_sw_8x8_3(%v144#1, %v146#0, %v132#3, %v158#2, %v290#3, %v291#2, %v302#1, %v303#0) {sym_name = "sw_11_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v146:8 = fabric.instance @__chess_sw_8x8_3(%v145#1, %v147#0, %v133#3, %v159#2, %v291#3, %v292#2, %v303#1, %v304#0) {sym_name = "sw_11_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v147:8 = fabric.instance @__chess_sw_8x8_3(%v146#1, %v148#0, %v134#3, %v160#2, %v292#3, %v293#2, %v304#1, %v305#0) {sym_name = "sw_11_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v148:8 = fabric.instance @__chess_sw_8x8_3(%v147#1, %v149#0, %v135#3, %v161#2, %v293#3, %v294#2, %v305#1, %v306#0) {sym_name = "sw_11_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v149:8 = fabric.instance @__chess_sw_8x8_3(%v148#1, %v150#0, %v136#3, %v162#2, %v294#3, %v295#2, %v306#1, %v307#0) {sym_name = "sw_11_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v150:8 = fabric.instance @__chess_sw_8x8_3(%v149#1, %v151#0, %v137#3, %v163#2, %v295#3, %v296#2, %v307#1, %v308#0) {sym_name = "sw_11_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v151:8 = fabric.instance @__chess_sw_8x8_3(%v150#1, %v152#0, %v138#3, %v164#2, %v296#3, %v297#2, %v308#1, %v309#0) {sym_name = "sw_11_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v152:8 = fabric.instance @__chess_sw_8x8_3(%v151#1, %v153#0, %v139#3, %v165#2, %v297#3, %v298#2, %v309#1, %v310#0) {sym_name = "sw_11_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v153:8 = fabric.instance @__chess_sw_8x8_3(%v152#1, %v154#0, %v140#3, %v166#2, %v298#3, %v299#2, %v310#1, %v311#0) {sym_name = "sw_11_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v154:8 = fabric.instance @__chess_sw_8x8_3(%v153#1, %v155#0, %v141#3, %v167#2, %v299#3, %v300#2, %v311#1, %v312#0) {sym_name = "sw_11_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v155:5 = fabric.instance @__chess_sw_5x5_1(%v154#1, %v142#2, %v168#1, %v300#3, %v312#1) {sym_name = "sw_11_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v156:6 = fabric.instance @__chess_sw_3x6_4(%v157#0, %v143#2, %v301#2) {sym_name = "sw_12_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v157:5 = fabric.instance @__chess_sw_5x5_1(%v156#0, %v158#0, %v144#3, %v301#3, %v302#2) {sym_name = "sw_12_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v158:5 = fabric.instance @__chess_sw_5x5_1(%v157#1, %v159#0, %v145#3, %v302#3, %v303#2) {sym_name = "sw_12_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v159:5 = fabric.instance @__chess_sw_5x5_1(%v158#1, %v160#0, %v146#3, %v303#3, %v304#2) {sym_name = "sw_12_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v160:5 = fabric.instance @__chess_sw_5x5_1(%v159#1, %v161#0, %v147#3, %v304#3, %v305#2) {sym_name = "sw_12_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v161:5 = fabric.instance @__chess_sw_5x5_1(%v160#1, %v162#0, %v148#3, %v305#3, %v306#2) {sym_name = "sw_12_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v162:5 = fabric.instance @__chess_sw_5x5_1(%v161#1, %v163#0, %v149#3, %v306#3, %v307#2) {sym_name = "sw_12_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v163:5 = fabric.instance @__chess_sw_5x5_1(%v162#1, %v164#0, %v150#3, %v307#3, %v308#2) {sym_name = "sw_12_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v164:5 = fabric.instance @__chess_sw_5x5_1(%v163#1, %v165#0, %v151#3, %v308#3, %v309#2) {sym_name = "sw_12_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v165:5 = fabric.instance @__chess_sw_5x5_1(%v164#1, %v166#0, %v152#3, %v309#3, %v310#2) {sym_name = "sw_12_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v166:5 = fabric.instance @__chess_sw_5x5_1(%v165#1, %v167#0, %v153#3, %v310#3, %v311#2) {sym_name = "sw_12_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v167:5 = fabric.instance @__chess_sw_5x5_1(%v166#1, %v168#0, %v154#3, %v311#3, %v312#2) {sym_name = "sw_12_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v168:8 = fabric.instance @__chess_sw_3x8_5(%v167#1, %v155#2, %v312#3) {sym_name = "sw_12_12"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v169:4 = fabric.instance @SC-FP_core_spe(%v0#2, %v1#3, %v13#3, %v14#4) {sym_name = "pe_0_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v170:4 = fabric.instance @SC-FP_core_spe(%v1#4, %v2#3, %v14#5, %v15#4) {sym_name = "pe_0_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v171:4 = fabric.instance @SC-FP_core_spe(%v2#4, %v3#3, %v15#5, %v16#4) {sym_name = "pe_0_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v172:4 = fabric.instance @SC-FP_core_spe(%v3#4, %v4#3, %v16#5, %v17#4) {sym_name = "pe_0_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v173:4 = fabric.instance @SC-FP_core_spe(%v4#4, %v5#3, %v17#5, %v18#4) {sym_name = "pe_0_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v174:4 = fabric.instance @SC-FP_core_spe(%v5#4, %v6#3, %v18#5, %v19#4) {sym_name = "pe_0_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v175:4 = fabric.instance @SC-FP_core_spe(%v6#4, %v7#3, %v19#5, %v20#4) {sym_name = "pe_0_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v176:4 = fabric.instance @SC-FP_core_spe(%v7#4, %v8#3, %v20#5, %v21#4) {sym_name = "pe_0_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v177:4 = fabric.instance @SC-FP_core_spe(%v8#4, %v9#3, %v21#5, %v22#4) {sym_name = "pe_0_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v178:4 = fabric.instance @SC-FP_core_spe(%v9#4, %v10#3, %v22#5, %v23#4) {sym_name = "pe_0_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v179:4 = fabric.instance @SC-FP_core_spe(%v10#4, %v11#3, %v23#5, %v24#4) {sym_name = "pe_0_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v180:4 = fabric.instance @SC-FP_core_spe(%v11#4, %v12#2, %v24#5, %v25#3) {sym_name = "pe_0_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v181:4 = fabric.instance @SC-FP_core_spe(%v13#4, %v14#6, %v26#3, %v27#4) {sym_name = "pe_1_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v182:4 = fabric.instance @SC-FP_core_spe(%v14#7, %v15#6, %v27#5, %v28#4) {sym_name = "pe_1_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v183:4 = fabric.instance @SC-FP_core_spe(%v15#7, %v16#6, %v28#5, %v29#4) {sym_name = "pe_1_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v184:4 = fabric.instance @SC-FP_core_spe(%v16#7, %v17#6, %v29#5, %v30#4) {sym_name = "pe_1_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v185:4 = fabric.instance @SC-FP_core_spe(%v17#7, %v18#6, %v30#5, %v31#4) {sym_name = "pe_1_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v186:4 = fabric.instance @SC-FP_core_spe(%v18#7, %v19#6, %v31#5, %v32#4) {sym_name = "pe_1_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v187:4 = fabric.instance @SC-FP_core_spe(%v19#7, %v20#6, %v32#5, %v33#4) {sym_name = "pe_1_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v188:4 = fabric.instance @SC-FP_core_spe(%v20#7, %v21#6, %v33#5, %v34#4) {sym_name = "pe_1_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v189:4 = fabric.instance @SC-FP_core_spe(%v21#7, %v22#6, %v34#5, %v35#4) {sym_name = "pe_1_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v190:4 = fabric.instance @SC-FP_core_spe(%v22#7, %v23#6, %v35#5, %v36#4) {sym_name = "pe_1_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v191:4 = fabric.instance @SC-FP_core_spe(%v23#7, %v24#6, %v36#5, %v37#4) {sym_name = "pe_1_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v192:4 = fabric.instance @SC-FP_core_spe(%v24#7, %v25#4, %v37#5, %v38#3) {sym_name = "pe_1_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v193:4 = fabric.instance @SC-FP_core_spe(%v26#4, %v27#6, %v39#3, %v40#4) {sym_name = "pe_2_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v194:4 = fabric.instance @SC-FP_core_spe(%v27#7, %v28#6, %v40#5, %v41#4) {sym_name = "pe_2_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v195:4 = fabric.instance @SC-FP_core_spe(%v28#7, %v29#6, %v41#5, %v42#4) {sym_name = "pe_2_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v196:4 = fabric.instance @SC-FP_core_spe(%v29#7, %v30#6, %v42#5, %v43#4) {sym_name = "pe_2_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v197:4 = fabric.instance @SC-FP_core_spe(%v30#7, %v31#6, %v43#5, %v44#4) {sym_name = "pe_2_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v198:4 = fabric.instance @SC-FP_core_spe(%v31#7, %v32#6, %v44#5, %v45#4) {sym_name = "pe_2_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v199:4 = fabric.instance @SC-FP_core_spe(%v32#7, %v33#6, %v45#5, %v46#4) {sym_name = "pe_2_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v200:4 = fabric.instance @SC-FP_core_spe(%v33#7, %v34#6, %v46#5, %v47#4) {sym_name = "pe_2_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v201:4 = fabric.instance @SC-FP_core_spe(%v34#7, %v35#6, %v47#5, %v48#4) {sym_name = "pe_2_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v202:4 = fabric.instance @SC-FP_core_spe(%v35#7, %v36#6, %v48#5, %v49#4) {sym_name = "pe_2_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v203:4 = fabric.instance @SC-FP_core_spe(%v36#7, %v37#6, %v49#5, %v50#4) {sym_name = "pe_2_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v204:4 = fabric.instance @SC-FP_core_spe(%v37#7, %v38#4, %v50#5, %v51#3) {sym_name = "pe_2_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v205:4 = fabric.instance @SC-FP_core_spe(%v39#4, %v40#6, %v52#3, %v53#4) {sym_name = "pe_3_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v206:4 = fabric.instance @SC-FP_core_spe(%v40#7, %v41#6, %v53#5, %v54#4) {sym_name = "pe_3_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v207:4 = fabric.instance @SC-FP_core_spe(%v41#7, %v42#6, %v54#5, %v55#4) {sym_name = "pe_3_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v208:4 = fabric.instance @SC-FP_core_spe(%v42#7, %v43#6, %v55#5, %v56#4) {sym_name = "pe_3_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v209:4 = fabric.instance @SC-FP_core_spe(%v43#7, %v44#6, %v56#5, %v57#4) {sym_name = "pe_3_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v210:4 = fabric.instance @SC-FP_core_spe(%v44#7, %v45#6, %v57#5, %v58#4) {sym_name = "pe_3_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v211:4 = fabric.instance @SC-FP_core_spe(%v45#7, %v46#6, %v58#5, %v59#4) {sym_name = "pe_3_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v212:4 = fabric.instance @SC-FP_core_spe(%v46#7, %v47#6, %v59#5, %v60#4) {sym_name = "pe_3_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v213:4 = fabric.instance @SC-FP_core_spe(%v47#7, %v48#6, %v60#5, %v61#4) {sym_name = "pe_3_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v214:4 = fabric.instance @SC-FP_core_spe(%v48#7, %v49#6, %v61#5, %v62#4) {sym_name = "pe_3_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v215:4 = fabric.instance @SC-FP_core_spe(%v49#7, %v50#6, %v62#5, %v63#4) {sym_name = "pe_3_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v216:4 = fabric.instance @SC-FP_core_spe(%v50#7, %v51#4, %v63#5, %v64#3) {sym_name = "pe_3_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v217:4 = fabric.instance @SC-FP_core_spe(%v52#4, %v53#6, %v65#3, %v66#4) {sym_name = "pe_4_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v218:4 = fabric.instance @SC-FP_core_spe(%v53#7, %v54#6, %v66#5, %v67#4) {sym_name = "pe_4_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v219:4 = fabric.instance @SC-FP_core_spe(%v54#7, %v55#6, %v67#5, %v68#4) {sym_name = "pe_4_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v220:4 = fabric.instance @SC-FP_core_spe(%v55#7, %v56#6, %v68#5, %v69#4) {sym_name = "pe_4_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v221:4 = fabric.instance @SC-FP_core_spe(%v56#7, %v57#6, %v69#5, %v70#4) {sym_name = "pe_4_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v222:4 = fabric.instance @SC-FP_core_spe(%v57#7, %v58#6, %v70#5, %v71#4) {sym_name = "pe_4_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v223:4 = fabric.instance @SC-FP_core_spe(%v58#7, %v59#6, %v71#5, %v72#4) {sym_name = "pe_4_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v224:4 = fabric.instance @SC-FP_core_spe(%v59#7, %v60#6, %v72#5, %v73#4) {sym_name = "pe_4_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v225:4 = fabric.instance @SC-FP_core_spe(%v60#7, %v61#6, %v73#5, %v74#4) {sym_name = "pe_4_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v226:4 = fabric.instance @SC-FP_core_spe(%v61#7, %v62#6, %v74#5, %v75#4) {sym_name = "pe_4_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v227:4 = fabric.instance @SC-FP_core_spe(%v62#7, %v63#6, %v75#5, %v76#4) {sym_name = "pe_4_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v228:4 = fabric.instance @SC-FP_core_spe(%v63#7, %v64#4, %v76#5, %v77#3) {sym_name = "pe_4_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v229:4 = fabric.instance @SC-FP_core_spe(%v65#4, %v66#6, %v78#3, %v79#4) {sym_name = "pe_5_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v230:4 = fabric.instance @SC-FP_core_spe(%v66#7, %v67#6, %v79#5, %v80#4) {sym_name = "pe_5_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v231:4 = fabric.instance @SC-FP_core_spe(%v67#7, %v68#6, %v80#5, %v81#4) {sym_name = "pe_5_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v232:4 = fabric.instance @SC-FP_core_spe(%v68#7, %v69#6, %v81#5, %v82#4) {sym_name = "pe_5_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v233:4 = fabric.instance @SC-FP_core_spe(%v69#7, %v70#6, %v82#5, %v83#4) {sym_name = "pe_5_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v234:4 = fabric.instance @SC-FP_core_spe(%v70#7, %v71#6, %v83#5, %v84#4) {sym_name = "pe_5_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v235:4 = fabric.instance @SC-FP_core_spe(%v71#7, %v72#6, %v84#5, %v85#4) {sym_name = "pe_5_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v236:4 = fabric.instance @SC-FP_core_spe(%v72#7, %v73#6, %v85#5, %v86#4) {sym_name = "pe_5_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v237:4 = fabric.instance @SC-FP_core_spe(%v73#7, %v74#6, %v86#5, %v87#4) {sym_name = "pe_5_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v238:4 = fabric.instance @SC-FP_core_spe(%v74#7, %v75#6, %v87#5, %v88#4) {sym_name = "pe_5_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v239:4 = fabric.instance @SC-FP_core_spe(%v75#7, %v76#6, %v88#5, %v89#4) {sym_name = "pe_5_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v240:4 = fabric.instance @SC-FP_core_spe(%v76#7, %v77#4, %v89#5, %v90#3) {sym_name = "pe_5_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v241:4 = fabric.instance @SC-FP_core_spe(%v78#4, %v79#6, %v91#3, %v92#4) {sym_name = "pe_6_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v242:4 = fabric.instance @SC-FP_core_spe(%v79#7, %v80#6, %v92#5, %v93#4) {sym_name = "pe_6_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v243:4 = fabric.instance @SC-FP_core_spe(%v80#7, %v81#6, %v93#5, %v94#4) {sym_name = "pe_6_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v244:4 = fabric.instance @SC-FP_core_spe(%v81#7, %v82#6, %v94#5, %v95#4) {sym_name = "pe_6_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v245:4 = fabric.instance @SC-FP_core_spe(%v82#7, %v83#6, %v95#5, %v96#4) {sym_name = "pe_6_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v246:4 = fabric.instance @SC-FP_core_spe(%v83#7, %v84#6, %v96#5, %v97#4) {sym_name = "pe_6_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v247:4 = fabric.instance @SC-FP_core_spe(%v84#7, %v85#6, %v97#5, %v98#4) {sym_name = "pe_6_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v248:4 = fabric.instance @SC-FP_core_spe(%v85#7, %v86#6, %v98#5, %v99#4) {sym_name = "pe_6_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v249:4 = fabric.instance @SC-FP_core_spe(%v86#7, %v87#6, %v99#5, %v100#4) {sym_name = "pe_6_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v250:4 = fabric.instance @SC-FP_core_spe(%v87#7, %v88#6, %v100#5, %v101#4) {sym_name = "pe_6_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v251:4 = fabric.instance @SC-FP_core_spe(%v88#7, %v89#6, %v101#5, %v102#4) {sym_name = "pe_6_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v252:4 = fabric.instance @SC-FP_core_spe(%v89#7, %v90#4, %v102#5, %v103#3) {sym_name = "pe_6_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v253:4 = fabric.instance @SC-FP_core_spe(%v91#4, %v92#6, %v104#3, %v105#4) {sym_name = "pe_7_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v254:4 = fabric.instance @SC-FP_core_spe(%v92#7, %v93#6, %v105#5, %v106#4) {sym_name = "pe_7_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v255:4 = fabric.instance @SC-FP_core_spe(%v93#7, %v94#6, %v106#5, %v107#4) {sym_name = "pe_7_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v256:4 = fabric.instance @SC-FP_core_spe(%v94#7, %v95#6, %v107#5, %v108#4) {sym_name = "pe_7_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v257:4 = fabric.instance @SC-FP_core_spe(%v95#7, %v96#6, %v108#5, %v109#4) {sym_name = "pe_7_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v258:4 = fabric.instance @SC-FP_core_spe(%v96#7, %v97#6, %v109#5, %v110#4) {sym_name = "pe_7_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v259:4 = fabric.instance @SC-FP_core_spe(%v97#7, %v98#6, %v110#5, %v111#4) {sym_name = "pe_7_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v260:4 = fabric.instance @SC-FP_core_spe(%v98#7, %v99#6, %v111#5, %v112#4) {sym_name = "pe_7_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v261:4 = fabric.instance @SC-FP_core_spe(%v99#7, %v100#6, %v112#5, %v113#4) {sym_name = "pe_7_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v262:4 = fabric.instance @SC-FP_core_spe(%v100#7, %v101#6, %v113#5, %v114#4) {sym_name = "pe_7_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v263:4 = fabric.instance @SC-FP_core_spe(%v101#7, %v102#6, %v114#5, %v115#4) {sym_name = "pe_7_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v264:4 = fabric.instance @SC-FP_core_spe(%v102#7, %v103#4, %v115#5, %v116#3) {sym_name = "pe_7_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v265:4 = fabric.instance @SC-FP_core_spe(%v104#4, %v105#6, %v117#3, %v118#4) {sym_name = "pe_8_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v266:4 = fabric.instance @SC-FP_core_spe(%v105#7, %v106#6, %v118#5, %v119#4) {sym_name = "pe_8_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v267:4 = fabric.instance @SC-FP_core_spe(%v106#7, %v107#6, %v119#5, %v120#4) {sym_name = "pe_8_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v268:4 = fabric.instance @SC-FP_core_spe(%v107#7, %v108#6, %v120#5, %v121#4) {sym_name = "pe_8_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v269:4 = fabric.instance @SC-FP_core_spe(%v108#7, %v109#6, %v121#5, %v122#4) {sym_name = "pe_8_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v270:4 = fabric.instance @SC-FP_core_spe(%v109#7, %v110#6, %v122#5, %v123#4) {sym_name = "pe_8_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v271:4 = fabric.instance @SC-FP_core_spe(%v110#7, %v111#6, %v123#5, %v124#4) {sym_name = "pe_8_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v272:4 = fabric.instance @SC-FP_core_spe(%v111#7, %v112#6, %v124#5, %v125#4) {sym_name = "pe_8_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v273:4 = fabric.instance @SC-FP_core_spe(%v112#7, %v113#6, %v125#5, %v126#4) {sym_name = "pe_8_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v274:4 = fabric.instance @SC-FP_core_spe(%v113#7, %v114#6, %v126#5, %v127#4) {sym_name = "pe_8_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v275:4 = fabric.instance @SC-FP_core_spe(%v114#7, %v115#6, %v127#5, %v128#4) {sym_name = "pe_8_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v276:4 = fabric.instance @SC-FP_core_spe(%v115#7, %v116#4, %v128#5, %v129#3) {sym_name = "pe_8_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v277:4 = fabric.instance @SC-FP_core_spe(%v117#4, %v118#6, %v130#3, %v131#4) {sym_name = "pe_9_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v278:4 = fabric.instance @SC-FP_core_spe(%v118#7, %v119#6, %v131#5, %v132#4) {sym_name = "pe_9_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v279:4 = fabric.instance @SC-FP_core_spe(%v119#7, %v120#6, %v132#5, %v133#4) {sym_name = "pe_9_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v280:4 = fabric.instance @SC-FP_core_spe(%v120#7, %v121#6, %v133#5, %v134#4) {sym_name = "pe_9_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v281:4 = fabric.instance @SC-FP_core_spe(%v121#7, %v122#6, %v134#5, %v135#4) {sym_name = "pe_9_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v282:4 = fabric.instance @SC-FP_core_spe(%v122#7, %v123#6, %v135#5, %v136#4) {sym_name = "pe_9_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v283:4 = fabric.instance @SC-FP_core_spe(%v123#7, %v124#6, %v136#5, %v137#4) {sym_name = "pe_9_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v284:4 = fabric.instance @SC-FP_core_spe(%v124#7, %v125#6, %v137#5, %v138#4) {sym_name = "pe_9_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v285:4 = fabric.instance @SC-FP_core_spe(%v125#7, %v126#6, %v138#5, %v139#4) {sym_name = "pe_9_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v286:4 = fabric.instance @SC-FP_core_spe(%v126#7, %v127#6, %v139#5, %v140#4) {sym_name = "pe_9_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v287:4 = fabric.instance @SC-FP_core_spe(%v127#7, %v128#6, %v140#5, %v141#4) {sym_name = "pe_9_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v288:4 = fabric.instance @SC-FP_core_spe(%v128#7, %v129#4, %v141#5, %v142#3) {sym_name = "pe_9_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v289:4 = fabric.instance @SC-FP_core_spe(%v130#4, %v131#6, %v143#3, %v144#4) {sym_name = "pe_10_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v290:4 = fabric.instance @SC-FP_core_spe(%v131#7, %v132#6, %v144#5, %v145#4) {sym_name = "pe_10_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v291:4 = fabric.instance @SC-FP_core_spe(%v132#7, %v133#6, %v145#5, %v146#4) {sym_name = "pe_10_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v292:4 = fabric.instance @SC-FP_core_spe(%v133#7, %v134#6, %v146#5, %v147#4) {sym_name = "pe_10_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v293:4 = fabric.instance @SC-FP_core_spe(%v134#7, %v135#6, %v147#5, %v148#4) {sym_name = "pe_10_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v294:4 = fabric.instance @SC-FP_core_spe(%v135#7, %v136#6, %v148#5, %v149#4) {sym_name = "pe_10_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v295:4 = fabric.instance @SC-FP_core_spe(%v136#7, %v137#6, %v149#5, %v150#4) {sym_name = "pe_10_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v296:4 = fabric.instance @SC-FP_core_spe(%v137#7, %v138#6, %v150#5, %v151#4) {sym_name = "pe_10_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v297:4 = fabric.instance @SC-FP_core_spe(%v138#7, %v139#6, %v151#5, %v152#4) {sym_name = "pe_10_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v298:4 = fabric.instance @SC-FP_core_spe(%v139#7, %v140#6, %v152#5, %v153#4) {sym_name = "pe_10_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v299:4 = fabric.instance @SC-FP_core_spe(%v140#7, %v141#6, %v153#5, %v154#4) {sym_name = "pe_10_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v300:4 = fabric.instance @SC-FP_core_spe(%v141#7, %v142#4, %v154#5, %v155#3) {sym_name = "pe_10_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v301:4 = fabric.instance @SC-FP_core_spe(%v143#4, %v144#6, %v156#2, %v157#3) {sym_name = "pe_11_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v302:4 = fabric.instance @SC-FP_core_spe(%v144#7, %v145#6, %v157#4, %v158#3) {sym_name = "pe_11_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v303:4 = fabric.instance @SC-FP_core_spe(%v145#7, %v146#6, %v158#4, %v159#3) {sym_name = "pe_11_2"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v304:4 = fabric.instance @SC-FP_core_spe(%v146#7, %v147#6, %v159#4, %v160#3) {sym_name = "pe_11_3"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v305:4 = fabric.instance @SC-FP_core_spe(%v147#7, %v148#6, %v160#4, %v161#3) {sym_name = "pe_11_4"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v306:4 = fabric.instance @SC-FP_core_spe(%v148#7, %v149#6, %v161#4, %v162#3) {sym_name = "pe_11_5"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v307:4 = fabric.instance @SC-FP_core_spe(%v149#7, %v150#6, %v162#4, %v163#3) {sym_name = "pe_11_6"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v308:4 = fabric.instance @SC-FP_core_spe(%v150#7, %v151#6, %v163#4, %v164#3) {sym_name = "pe_11_7"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v309:4 = fabric.instance @SC-FP_core_spe(%v151#7, %v152#6, %v164#4, %v165#3) {sym_name = "pe_11_8"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v310:4 = fabric.instance @SC-FP_core_spe(%v152#7, %v153#6, %v165#4, %v166#3) {sym_name = "pe_11_9"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v311:4 = fabric.instance @SC-FP_core_spe(%v153#7, %v154#6, %v166#4, %v167#3) {sym_name = "pe_11_10"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v312:4 = fabric.instance @SC-FP_core_spe(%v154#7, %v155#4, %v167#4, %v168#2) {sym_name = "pe_11_11"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v313:3 = fabric.extmemory @extmem_0 [ldCount = 2, stCount = 1, lsqDepth = 0, memrefType = memref<?xi64>] (%mem0, %v156#3, %v156#4, %v156#5) : (memref<?xi64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %v314:3 = fabric.extmemory @extmem_1 [ldCount = 2, stCount = 1, lsqDepth = 0, memrefType = memref<?xi64>] (%mem1, %v168#3, %v168#4, %v168#5) : (memref<?xi64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.yield %v168#6, %v168#7 : !fabric.bits<64>, !fabric.bits<64>
}
}
// CORE_TYPE_METADATA
// spm_capacity_bytes = 32768
