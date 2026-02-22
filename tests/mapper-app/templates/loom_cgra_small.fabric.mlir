// Small CGRA template (4x4 grid equivalent)
// Targets apps with <= 300 lines in handshake.mlir
//
// Resources: heterogeneous PE tile array with control flow,
//   constant, arithmetic, dataflow, and memory support.
// Architecture: layered mesh with typed crossbar routing.

module {

// ---------------------------------------------------------------------------
// Constant PEs (handshake.constant for various types)
// ---------------------------------------------------------------------------
fabric.pe @pe_const_i32(%ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = handshake.constant %ctrl {value = 0 : i32} : i32
  fabric.yield %r : i32
}
fabric.pe @pe_const_i64(%ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = handshake.constant %ctrl {value = 0 : i64} : i64
  fabric.yield %r : i64
}
fabric.pe @pe_const_index(%ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = handshake.constant %ctrl {value = 0 : index} : index
  fabric.yield %r : index
}
fabric.pe @pe_const_i1(%ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = handshake.constant %ctrl {value = true} : i1
  fabric.yield %r : i1
}

// ---------------------------------------------------------------------------
// Control flow PEs (join, cond_br, mux)
// ---------------------------------------------------------------------------
fabric.pe @pe_join1(%a: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.join %a : none
  fabric.yield %r : none
}
fabric.pe @pe_join2(%a: none, %b: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.join %a, %b : none, none
  fabric.yield %r : none
}
fabric.pe @pe_join3(%a: none, %b: none, %c: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.join %a, %b, %c : none, none, none
  fabric.yield %r : none
}
fabric.pe @pe_join4(%a: none, %b: none, %c: none, %d: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.join %a, %b, %c, %d : none, none, none, none
  fabric.yield %r : none
}
fabric.pe @pe_cond_br_none(%cond: i1, %data: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none, none) {
  %t, %f = handshake.cond_br %cond, %data : none
  fabric.yield %t, %f : none, none
}
fabric.pe @pe_cond_br_i32(%cond: i1, %data: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, i32) {
  %t, %f = handshake.cond_br %cond, %data : i32
  fabric.yield %t, %f : i32, i32
}
fabric.pe @pe_mux_i32(%sel: index, %a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = handshake.mux %sel [%a, %b] : index, i32
  fabric.yield %r : i32
}
fabric.pe @pe_mux_none(%sel: index, %a: none, %b: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.mux %sel [%a, %b] : index, none
  fabric.yield %r : none
}

// ---------------------------------------------------------------------------
// Integer arithmetic PEs
// ---------------------------------------------------------------------------
fabric.pe @pe_addi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.addi %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_subi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.subi %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_muli(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 2 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.muli %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_andi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.andi %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_ori(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.ori %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_xori(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.xori %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_shli(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.shli %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_shrui(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.shrui %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_shrsi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.shrsi %a, %b : i32
  fabric.yield %r : i32
}

// ---------------------------------------------------------------------------
// Compare, select, type conversion PEs
// ---------------------------------------------------------------------------
fabric.pe @pe_cmpi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpi ult, %a, %b : i32
  fabric.yield %r : i1
}
fabric.pe @pe_select(%c: i1, %a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.select %c, %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_select_index(%c: i1, %a: index, %b: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.select %c, %a, %b : index
  fabric.yield %r : index
}
fabric.pe @pe_index_cast_i64(%v: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.index_cast %v : i64 to index
  fabric.yield %r : index
}
fabric.pe @pe_index_cast_i32(%v: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.index_cast %v : i32 to index
  fabric.yield %r : index
}

// ---------------------------------------------------------------------------
// Load and store PEs
// ---------------------------------------------------------------------------
fabric.pe @pe_load(%addr: index, %data: i32, %ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, index) {
  %ld_d, %ld_a = handshake.load [%addr] %data, %ctrl : index, i32
  fabric.yield %ld_d, %ld_a : i32, index
}
fabric.pe @pe_store(%addr: index, %data: i32, %ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, index) {
  %st_d, %st_a = handshake.store [%addr] %data, %ctrl : index, i32
  fabric.yield %st_d, %st_a : i32, index
}

// ---------------------------------------------------------------------------
// Dataflow PEs
// ---------------------------------------------------------------------------
fabric.pe @pe_invariant(%d: i1, %a: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %o = dataflow.invariant %d, %a : i1, i32 -> i32
  fabric.yield %o : i32
}
fabric.pe @pe_carry(%d: i1, %a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %o = dataflow.carry %d, %a, %b : i1, i32, i32 -> i32
  fabric.yield %o : i32
}
fabric.pe @pe_carry_none(%d: i1, %a: none, %b: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %o = dataflow.carry %d, %a, %b : i1, none, none -> none
  fabric.yield %o : none
}
fabric.pe @pe_gate(%val: i32, %cond: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, i1) {
  %av, %ac = dataflow.gate %val, %cond : i32, i1 -> i32, i1
  fabric.yield %av, %ac : i32, i1
}
fabric.pe @pe_gate_index(%val: index, %cond: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index, i1) {
  %av, %ac = dataflow.gate %val, %cond : index, i1 -> index, i1
  fabric.yield %av, %ac : index, i1
}
fabric.pe @pe_stream(%start: index, %step: index, %bound: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index, i1) {
  %idx, %wc = dataflow.stream %start, %step, %bound
  fabric.yield %idx, %wc : index, i1
}

// Sink PEs
fabric.pe @pe_sink_i1(%v: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> () {
  handshake.sink %v : i1
  fabric.yield
}
fabric.pe @pe_sink_none(%v: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> () {
  handshake.sink %v : none
  fabric.yield
}

// ---------------------------------------------------------------------------
// Switches (typed crossbars for routing)
// ---------------------------------------------------------------------------
fabric.switch @sw4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
fabric.switch @sw_none4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (none, none, none, none) -> (none, none, none, none)
fabric.switch @sw_index4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (index, index, index, index) -> (index, index, index, index)
fabric.switch @sw_i1_4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (i1, i1, i1, i1) -> (i1, i1, i1, i1)
fabric.fifo @fifo_buf [depth = 2] : (i32) -> (i32)

// ---------------------------------------------------------------------------
// Top-level module: layered heterogeneous CGRA
// Each PE output has exactly 1 consumer (inline broadcast for fanout).
// ---------------------------------------------------------------------------
fabric.module @loom_cgra_small(
    %mem0: memref<?xi32, strided<[1], offset: ?>>,
    %ctrl_in: none,
    %in0: i32, %in1: i32, %in2: i32, %in3: i32,
    %addr0: index
) -> (i32, i32) {

  // ===== Layer 0: Input broadcast =====
  // Control token broadcast tree (none)
  %cn0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %ctrl_in : none -> none, none, none, none
  %cn00:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cn0#0 : none -> none, none, none, none
  %cn01:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cn0#1 : none -> none, none, none, none
  %cn02:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cn0#2 : none -> none, none, none, none
  %cn03:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cn0#3 : none -> none, none, none, none

  // Data input broadcast (i32)
  %d0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in0 : i32 -> i32, i32, i32, i32
  %d1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in1 : i32 -> i32, i32, i32, i32
  %d2:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in2 : i32 -> i32, i32, i32, i32
  %d3:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in3 : i32 -> i32, i32, i32, i32

  // Address input broadcast (index)
  %a0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %addr0 : index -> index, index, index, index

  // ===== Layer 1: Constant + Join PEs =====
  %ci32_0 = fabric.instance @pe_const_i32(%cn00#0) {sym_name = "ci32_0"} : (none) -> i32
  %ci32_1 = fabric.instance @pe_const_i32(%cn00#1) {sym_name = "ci32_1"} : (none) -> i32
  %ci32_2 = fabric.instance @pe_const_i32(%cn00#2) {sym_name = "ci32_2"} : (none) -> i32
  %ci32_3 = fabric.instance @pe_const_i32(%cn00#3) {sym_name = "ci32_3"} : (none) -> i32
  %ci64_0 = fabric.instance @pe_const_i64(%cn01#0) {sym_name = "ci64_0"} : (none) -> i64
  %ci64_1 = fabric.instance @pe_const_i64(%cn01#1) {sym_name = "ci64_1"} : (none) -> i64
  %cidx_0 = fabric.instance @pe_const_index(%cn01#2) {sym_name = "cidx_0"} : (none) -> index
  %cidx_1 = fabric.instance @pe_const_index(%cn01#3) {sym_name = "cidx_1"} : (none) -> index
  %cidx_2 = fabric.instance @pe_const_index(%cn02#0) {sym_name = "cidx_2"} : (none) -> index
  %cidx_3 = fabric.instance @pe_const_index(%cn02#1) {sym_name = "cidx_3"} : (none) -> index
  %ci1_0 = fabric.instance @pe_const_i1(%cn02#2) {sym_name = "ci1_0"} : (none) -> i1
  %ci1_1 = fabric.instance @pe_const_i1(%cn02#3) {sym_name = "ci1_1"} : (none) -> i1
  %join_0 = fabric.instance @pe_join1(%cn03#0) {sym_name = "join_0"} : (none) -> none
  %join_1 = fabric.instance @pe_join1(%cn03#1) {sym_name = "join_1"} : (none) -> none
  %join2_0 = fabric.instance @pe_join2(%cn03#2, %cn03#3) {sym_name = "join2_0"} : (none, none) -> none

  // ===== Layer 2a: i32 routing crossbar =====
  %ri32_0:4 = fabric.instance @sw4x4(%ci32_0, %ci32_1, %fb_i32#0, %fb_i32#1)
    {sym_name = "ri32_0"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %ri32_1:4 = fabric.instance @sw4x4(%ci32_2, %ci32_3, %d2#0, %d3#0)
    {sym_name = "ri32_1"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // ===== Layer 2b: i64 routing =====
  %ri64:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %ci64_0 : i64 -> i64, i64, i64, i64

  // ===== Layer 2c: index routing =====
  %ridx_0:4 = fabric.instance @sw_index4x4(%cidx_0, %cidx_1, %fb_idx#0, %fb_idx#1)
    {sym_name = "ridx_0"} : (index, index, index, index) -> (index, index, index, index)

  // ===== Layer 3: Compute PEs =====
  // Arithmetic
  %add0 = fabric.instance @pe_addi(%ri32_0#0, %ri32_0#1) {sym_name = "addi_0"} : (i32, i32) -> i32
  %add1 = fabric.instance @pe_addi(%ri32_0#2, %ri32_0#3) {sym_name = "addi_1"} : (i32, i32) -> i32
  %sub0 = fabric.instance @pe_subi(%ri32_1#0, %ri32_1#1) {sym_name = "subi_0"} : (i32, i32) -> i32
  %mul0 = fabric.instance @pe_muli(%ri32_1#2, %ri32_1#3) {sym_name = "muli_0"} : (i32, i32) -> i32

  // Compare
  %cmp0 = fabric.instance @pe_cmpi(%d0#1, %d1#1) {sym_name = "cmpi_0"} : (i32, i32) -> i1
  %cmp1 = fabric.instance @pe_cmpi(%d2#1, %d3#1) {sym_name = "cmpi_1"} : (i32, i32) -> i1

  // Type conversion
  %icast64_0 = fabric.instance @pe_index_cast_i64(%ri64#0) {sym_name = "icast64_0"} : (i64) -> index
  %icast64_1 = fabric.instance @pe_index_cast_i64(%ri64#1) {sym_name = "icast64_1"} : (i64) -> index
  %icast32_0 = fabric.instance @pe_index_cast_i32(%d0#2) {sym_name = "icast32_0"} : (i32) -> index
  %icast32_1 = fabric.instance @pe_index_cast_i32(%d1#2) {sym_name = "icast32_1"} : (i32) -> index

  // ===== Layer 3 output broadcast =====
  %bc_cmp0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cmp0 : i1 -> i1, i1, i1, i1
  %bc_cmp1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cmp1 : i1 -> i1, i1, i1, i1

  // ===== Layer 4a: i1 routing =====
  %ri1_0:4 = fabric.instance @sw_i1_4x4(%bc_cmp0#0, %bc_cmp0#1, %bc_cmp1#0, %bc_cmp1#1)
    {sym_name = "ri1_0"} : (i1, i1, i1, i1) -> (i1, i1, i1, i1)
  %ri1_1:4 = fabric.instance @sw_i1_4x4(%fb_i1#0, %fb_i1#1, %ci1_0, %ci1_1)
    {sym_name = "ri1_1"} : (i1, i1, i1, i1) -> (i1, i1, i1, i1)

  // ===== Layer 4b: index routing 2 =====
  %ridx_1:4 = fabric.instance @sw_index4x4(%cidx_2, %cidx_3, %icast64_0, %icast32_0)
    {sym_name = "ridx_1"} : (index, index, index, index) -> (index, index, index, index)
  %ridx_2:4 = fabric.instance @sw_index4x4(%icast64_1, %icast32_1, %a0#2, %a0#3)
    {sym_name = "ridx_2"} : (index, index, index, index) -> (index, index, index, index)

  // ===== Layer 5: Select + Dataflow PEs =====
  %sel0 = fabric.instance @pe_select(%ri1_0#0, %d0#3, %d1#3) {sym_name = "sel_0"} : (i1, i32, i32) -> i32
  %sel1 = fabric.instance @pe_select(%ri1_0#1, %d2#2, %d3#2) {sym_name = "sel_1"} : (i1, i32, i32) -> i32
  %selidx0 = fabric.instance @pe_select_index(%ri1_0#2, %ridx_0#0, %ridx_0#1) {sym_name = "selidx_0"} : (i1, index, index) -> index
  %selidx1 = fabric.instance @pe_select_index(%ri1_0#3, %ridx_0#2, %ridx_0#3) {sym_name = "selidx_1"} : (i1, index, index) -> index

  // Dataflow stream
  %stream0:2 = fabric.instance @pe_stream(%ridx_1#0, %ridx_1#1, %ridx_1#2)
    {sym_name = "stream_0"} : (index, index, index) -> (index, i1)
  %stream1:2 = fabric.instance @pe_stream(%ridx_1#3, %ridx_2#0, %ridx_2#1)
    {sym_name = "stream_1"} : (index, index, index) -> (index, i1)

  // Broadcast stream outputs
  %bc_sidx0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %stream0#0 : index -> index, index, index, index
  %bc_swc0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %stream0#1 : i1 -> i1, i1, i1, i1
  %bc_sidx1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %stream1#0 : index -> index, index, index, index
  %bc_swc1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %stream1#1 : i1 -> i1, i1, i1, i1

  // Dataflow gate
  %gate_idx0:2 = fabric.instance @pe_gate_index(%bc_sidx0#0, %bc_swc0#0) {sym_name = "gate_idx_0"} : (index, i1) -> (index, i1)
  %gate_i32_0:2 = fabric.instance @pe_gate(%add0, %bc_swc0#1) {sym_name = "gate_i32_0"} : (i32, i1) -> (i32, i1)
  %gate_idx1:2 = fabric.instance @pe_gate_index(%bc_sidx1#0, %bc_swc1#0) {sym_name = "gate_idx_1"} : (index, i1) -> (index, i1)
  %gate_i32_1:2 = fabric.instance @pe_gate(%add1, %bc_swc1#1) {sym_name = "gate_i32_1"} : (i32, i1) -> (i32, i1)

  // Dataflow carry
  %carry0 = fabric.instance @pe_carry(%bc_swc0#2, %sel0, %sub0) {sym_name = "carry_0"} : (i1, i32, i32) -> i32
  %carry1 = fabric.instance @pe_carry(%bc_swc1#2, %sel1, %mul0) {sym_name = "carry_1"} : (i1, i32, i32) -> i32

  // ===== Layer 5b: none routing =====
  %bc_join0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %join_0 : none -> none, none, none, none
  %bc_join1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %join_1 : none -> none, none, none, none
  %bc_join2:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %join2_0 : none -> none, none, none, none

  %rnone_0:4 = fabric.instance @sw_none4x4(%fb_none#0, %fb_none#1, %bc_join1#0, %bc_join1#1)
    {sym_name = "rnone_0"} : (none, none, none, none) -> (none, none, none, none)
  %rnone_1:4 = fabric.instance @sw_none4x4(%bc_join0#2, %bc_join0#3, %bc_join2#0, %bc_join2#1)
    {sym_name = "rnone_1"} : (none, none, none, none) -> (none, none, none, none)

  // ===== Layer 6: Control flow PEs =====
  %cbr_n0:2 = fabric.instance @pe_cond_br_none(%ri1_1#0, %rnone_0#0) {sym_name = "cbr_n_0"} : (i1, none) -> (none, none)
  %cbr_n1:2 = fabric.instance @pe_cond_br_none(%ri1_1#1, %rnone_0#1) {sym_name = "cbr_n_1"} : (i1, none) -> (none, none)
  %cbr_n2:2 = fabric.instance @pe_cond_br_none(%bc_swc0#3, %rnone_0#2) {sym_name = "cbr_n_2"} : (i1, none) -> (none, none)
  %cbr_i0:2 = fabric.instance @pe_cond_br_i32(%gate_i32_0#1, %carry0) {sym_name = "cbr_i_0"} : (i1, i32) -> (i32, i32)
  %cbr_i1:2 = fabric.instance @pe_cond_br_i32(%gate_i32_1#1, %carry1) {sym_name = "cbr_i_1"} : (i1, i32) -> (i32, i32)

  // Carry for none type
  %bc_cbr_n0_t:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cbr_n0#0 : none -> none, none, none, none
  %bc_cbr_n0_f:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %cbr_n0#1 : none -> none, none, none, none
  %carry_n0 = fabric.instance @pe_carry_none(%ri1_1#2, %bc_cbr_n0_t#0, %bc_cbr_n0_f#0) {sym_name = "carry_n_0"} : (i1, none, none) -> none
  %carry_n1 = fabric.instance @pe_carry_none(%ri1_1#3, %bc_cbr_n0_t#1, %bc_cbr_n0_f#1) {sym_name = "carry_n_1"} : (i1, none, none) -> none

  // Broadcast selidx for mux
  %bc_selidx0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %selidx0 : index -> index, index, index, index
  %bc_selidx1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %selidx1 : index -> index, index, index, index

  // Mux PEs
  %mux_i0 = fabric.instance @pe_mux_i32(%bc_selidx0#0, %cbr_i0#0, %cbr_i0#1) {sym_name = "mux_i_0"} : (index, i32, i32) -> i32
  %mux_i1 = fabric.instance @pe_mux_i32(%bc_selidx0#1, %cbr_i1#0, %cbr_i1#1) {sym_name = "mux_i_1"} : (index, i32, i32) -> i32
  %mux_n0 = fabric.instance @pe_mux_none(%bc_selidx0#2, %cbr_n1#0, %cbr_n1#1) {sym_name = "mux_n_0"} : (index, none, none) -> none
  %mux_n1 = fabric.instance @pe_mux_none(%bc_selidx0#3, %cbr_n2#0, %cbr_n2#1) {sym_name = "mux_n_1"} : (index, none, none) -> none

  // ===== Layer 7: Memory + final routing =====
  // Load/Store PEs
  %load0:2 = fabric.instance @pe_load(%gate_idx0#0, %d2#3, %rnone_0#3) {sym_name = "load_0"} : (index, i32, none) -> (i32, index)
  %load1:2 = fabric.instance @pe_load(%gate_idx1#0, %d3#3, %rnone_1#0) {sym_name = "load_1"} : (index, i32, none) -> (i32, index)
  %store0:2 = fabric.instance @pe_store(%bc_sidx0#1, %mux_i0, %rnone_1#1) {sym_name = "store_0"} : (index, i32, none) -> (i32, index)

  // Invariant PE
  %inv0 = fabric.instance @pe_invariant(%bc_swc1#3, %mux_i1) {sym_name = "inv_0"} : (i1, i32) -> i32

  // Extra compute PEs (second pass)
  %add2 = fabric.instance @pe_addi(%load0#0, %gate_i32_0#0) {sym_name = "addi_2"} : (i32, i32) -> i32
  %add3 = fabric.instance @pe_addi(%load1#0, %gate_i32_1#0) {sym_name = "addi_3"} : (i32, i32) -> i32

  // External memory
  %ld_ext, %done_ext = fabric.extmemory
    [ldCount = 1, stCount = 1, lsqDepth = 4]
    (%mem0, %ridx_2#2)
    : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, index) -> (i32, none)

  // More none routing for output stage
  %rnone_2:4 = fabric.instance @sw_none4x4(%carry_n0, %carry_n1, %mux_n0, %mux_n1)
    {sym_name = "rnone_2"} : (none, none, none, none) -> (none, none, none, none)
  %rnone_3:4 = fabric.instance @sw_none4x4(%bc_cbr_n0_t#2, %bc_cbr_n0_f#2, %bc_join2#2, %done_ext)
    {sym_name = "rnone_3"} : (none, none, none, none) -> (none, none, none, none)

  // Final join PE variants
  %join3_0 = fabric.instance @pe_join3(%rnone_2#0, %rnone_2#1, %rnone_2#2) {sym_name = "join3_0"} : (none, none, none) -> none
  %join4_0 = fabric.instance @pe_join4(%rnone_3#0, %rnone_3#1, %rnone_3#2, %rnone_3#3) {sym_name = "join4_0"} : (none, none, none, none) -> none

  // Final index routing
  %ridx_3:4 = fabric.instance @sw_index4x4(%bc_sidx0#2, %bc_sidx0#3, %bc_sidx1#1, %load0#1)
    {sym_name = "ridx_3"} : (index, index, index, index) -> (index, index, index, index)
  %ridx_4:4 = fabric.instance @sw_index4x4(%bc_sidx1#2, %bc_sidx1#3, %load1#1, %store0#1)
    {sym_name = "ridx_4"} : (index, index, index, index) -> (index, index, index, index)
  %ridx_5:4 = fabric.instance @sw_index4x4(%bc_selidx1#0, %bc_selidx1#1, %bc_selidx1#2, %ridx_2#3)
    {sym_name = "ridx_5"} : (index, index, index, index) -> (index, index, index, index)

  // Output i32 routing
  %rout_0:4 = fabric.instance @sw4x4(%add2, %add3, %inv0, %store0#0)
    {sym_name = "rout_0"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Feedback routing switches (backward paths via forward references)
  %fb_i32:4 = fabric.instance @sw4x4(%rout_0#2, %rout_0#3, %d0#0, %d1#0)
    {sym_name = "fb_i32"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %fb_idx:4 = fabric.instance @sw_index4x4(%ridx_3#0, %ridx_4#0, %a0#0, %a0#1)
    {sym_name = "fb_idx"} : (index, index, index, index) -> (index, index, index, index)
  %fb_i1:4 = fabric.instance @sw_i1_4x4(%gate_idx0#1, %gate_idx1#1, %bc_cmp0#2, %bc_cmp0#3)
    {sym_name = "fb_i1"} : (i1, i1, i1, i1) -> (i1, i1, i1, i1)
  %fb_none:4 = fabric.instance @sw_none4x4(%rnone_2#3, %bc_cbr_n0_t#3, %bc_join0#0, %bc_join0#1)
    {sym_name = "fb_none"} : (none, none, none, none) -> (none, none, none, none)

  fabric.yield %rout_0#0, %rout_0#1 : i32, i32
}

}
