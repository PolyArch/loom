// Medium CGRA template (8x8 grid equivalent)
// Targets apps with 301-800 lines in handshake.mlir
//
// Resources: 16 compute PEs, 16 constants, 8 load + 8 store PEs,
//   4 private + 4 external memory, 8 dataflow PEs per type, float support

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
// Integer arithmetic PE definitions
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
// Float arithmetic PE definitions
// ---------------------------------------------------------------------------
fabric.pe @pe_addf(%a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 3 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.addf %a, %b : f32
  fabric.yield %r : f32
}
fabric.pe @pe_subf(%a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 3 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.subf %a, %b : f32
  fabric.yield %r : f32
}
fabric.pe @pe_mulf(%a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 3 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.mulf %a, %b : f32
  fabric.yield %r : f32
}
fabric.pe @pe_divf(%a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.divf %a, %b : f32
  fabric.yield %r : f32
}
fabric.pe @pe_negf(%a: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.negf %a : f32
  fabric.yield %r : f32
}

// ---------------------------------------------------------------------------
// Compare and select PEs
// ---------------------------------------------------------------------------
fabric.pe @pe_cmpi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpi ult, %a, %b : i32
  fabric.yield %r : i1
}
fabric.pe @pe_cmpf(%a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpf ult, %a, %b : f32
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

// ---------------------------------------------------------------------------
// Type cast PEs
// ---------------------------------------------------------------------------
fabric.pe @pe_extui(%a: i16) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.extui %a : i16 to i32
  fabric.yield %r : i32
}
fabric.pe @pe_extsi(%a: i16) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.extsi %a : i16 to i32
  fabric.yield %r : i32
}
fabric.pe @pe_trunci(%a: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i16) {
  %r = arith.trunci %a : i32 to i16
  fabric.yield %r : i16
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

// ---------------------------------------------------------------------------
// Switches and FIFOs
// ---------------------------------------------------------------------------
fabric.switch @sw4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
fabric.switch @sw_none4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (none, none, none, none) -> (none, none, none, none)
fabric.switch @sw_index4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (index, index, index, index) -> (index, index, index, index)
fabric.switch @sw_i1_4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (i1, i1, i1, i1) -> (i1, i1, i1, i1)
fabric.fifo @fifo_buf [depth = 2] : (i32) -> (i32)

// ---------------------------------------------------------------------------
// Top-level module: expanded 4x4 compute fabric
// ---------------------------------------------------------------------------
fabric.module @loom_cgra_medium(
    %mem0: memref<?xi32>,
    %mem1: memref<?xi32>,
    %in0: i32, %in1: i32, %in2: i32, %in3: i32,
    %in4: i32, %in5: i32, %in6: i32, %in7: i32,
    %addr0: index, %addr1: index
) -> (i32, i32, i32, i32) {
  // Input broadcast (strict 1-to-1 fanout)
  %b0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in0 : i32 -> i32, i32, i32, i32
  %b1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in1 : i32 -> i32, i32, i32, i32
  %b2:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in2 : i32 -> i32, i32, i32, i32
  %b3:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in3 : i32 -> i32, i32, i32, i32
  %b4:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in4 : i32 -> i32, i32, i32, i32
  %b5:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in5 : i32 -> i32, i32, i32, i32
  %b6:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in6 : i32 -> i32, i32, i32, i32
  %b7:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in7 : i32 -> i32, i32, i32, i32

  // Row 0: PEs + routing
  %r0a = fabric.instance @pe_addi(%b0#0, %b1#0) {sym_name = "pe_r0_addi"} : (i32, i32) -> i32
  %r0b = fabric.instance @pe_muli(%b2#0, %b3#0) {sym_name = "pe_r0_muli"} : (i32, i32) -> i32
  %r0c = fabric.instance @pe_subi(%b4#0, %b5#0) {sym_name = "pe_r0_subi"} : (i32, i32) -> i32
  %r0d = fabric.instance @pe_andi(%b6#0, %b7#0) {sym_name = "pe_r0_andi"} : (i32, i32) -> i32
  %sw_r0:4 = fabric.instance @sw4x4(%r0a, %r0b, %r0c, %r0d)
    {sym_name = "sw_r0"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Row 1: PEs + routing
  %r1a = fabric.instance @pe_addi(%b0#1, %b1#1) {sym_name = "pe_r1_addi"} : (i32, i32) -> i32
  %r1b = fabric.instance @pe_muli(%b2#1, %b3#1) {sym_name = "pe_r1_muli"} : (i32, i32) -> i32
  %r1c = fabric.instance @pe_ori(%b4#1, %b5#1) {sym_name = "pe_r1_ori"} : (i32, i32) -> i32
  %r1d = fabric.instance @pe_xori(%b6#1, %b7#1) {sym_name = "pe_r1_xori"} : (i32, i32) -> i32
  %sw_r1:4 = fabric.instance @sw4x4(%r1a, %r1b, %r1c, %r1d)
    {sym_name = "sw_r1"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Row 2: PEs + routing (cross-row connections)
  %r2a = fabric.instance @pe_addi(%sw_r0#0, %sw_r1#0) {sym_name = "pe_r2_addi"} : (i32, i32) -> i32
  %r2b = fabric.instance @pe_muli(%sw_r0#1, %sw_r1#1) {sym_name = "pe_r2_muli"} : (i32, i32) -> i32
  %r2c = fabric.instance @pe_subi(%sw_r0#2, %sw_r1#2) {sym_name = "pe_r2_subi"} : (i32, i32) -> i32
  %r2d = fabric.instance @pe_shli(%sw_r0#3, %sw_r1#3) {sym_name = "pe_r2_shli"} : (i32, i32) -> i32
  %sw_r2:4 = fabric.instance @sw4x4(%r2a, %r2b, %r2c, %r2d)
    {sym_name = "sw_r2"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Row 3: PEs + routing
  %r3a = fabric.instance @pe_addi(%sw_r2#0, %b0#2) {sym_name = "pe_r3_addi"} : (i32, i32) -> i32
  %r3b = fabric.instance @pe_muli(%sw_r2#1, %b1#2) {sym_name = "pe_r3_muli"} : (i32, i32) -> i32
  %r3c = fabric.instance @pe_shrui(%sw_r2#2, %b2#2) {sym_name = "pe_r3_shrui"} : (i32, i32) -> i32
  %r3d = fabric.instance @pe_shrsi(%sw_r2#3, %b3#2) {sym_name = "pe_r3_shrsi"} : (i32, i32) -> i32
  %sw_r3:4 = fabric.instance @sw4x4(%r3a, %r3b, %r3c, %r3d)
    {sym_name = "sw_r3"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // External memory
  %ld0, %done0 = fabric.extmemory
    [ldCount = 1, stCount = 0]
    (%mem0, %addr0)
    : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)
  %ld1, %done1 = fabric.extmemory
    [ldCount = 1, stCount = 0]
    (%mem1, %addr1)
    : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)

  fabric.yield %sw_r3#0, %sw_r3#1, %sw_r3#2, %sw_r3#3 : i32, i32, i32, i32
}

}
