// Large CGRA template (16x16 grid equivalent)
// Targets apps with > 800 lines in handshake.mlir
//
// Resources: 64 compute PEs, 32 constants, 16 load + 16 store PEs,
//   8 private + 8 external memory, 16 dataflow PEs per type, full op coverage

module {

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
fabric.pe @pe_divsi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.divsi %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_divui(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.divui %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_remsi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.remsi %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_remui(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.remui %a, %b : i32
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
fabric.pe @pe_uitofp(%a: i32) [latency = [1 : i16, 1 : i16, 3 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.uitofp %a : i32 to f32
  fabric.yield %r : f32
}
fabric.pe @pe_fptoui(%a: f32) [latency = [1 : i16, 1 : i16, 3 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.fptoui %a : f32 to i32
  fabric.yield %r : i32
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
fabric.pe @pe_gate(%val: i32, %cond: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, i1) {
  %av, %ac = dataflow.gate %val, %cond : i32, i1 -> i32, i1
  fabric.yield %av, %ac : i32, i1
}
fabric.pe @pe_stream(%start: index, %step: index, %bound: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index, i1) {
  %idx, %wc = dataflow.stream %start, %step, %bound
  fabric.yield %idx, %wc : index, i1
}

// ---------------------------------------------------------------------------
// Switches and FIFOs
// ---------------------------------------------------------------------------
fabric.switch @sw4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
fabric.fifo @fifo_buf [depth = 2] : (i32) -> (i32)

// ---------------------------------------------------------------------------
// Top-level module: 8-row deep compute fabric
// ---------------------------------------------------------------------------
fabric.module @loom_cgra_large(
    %mem0: memref<?xi32>,
    %mem1: memref<?xi32>,
    %mem2: memref<?xi32>,
    %mem3: memref<?xi32>,
    %in0: i32, %in1: i32, %in2: i32, %in3: i32,
    %in4: i32, %in5: i32, %in6: i32, %in7: i32,
    %in8: i32, %in9: i32, %in10: i32, %in11: i32,
    %in12: i32, %in13: i32, %in14: i32, %in15: i32,
    %addr0: index, %addr1: index, %addr2: index, %addr3: index
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
  // Input broadcast layer
  %b0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in0 : i32 -> i32, i32, i32, i32
  %b1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in1 : i32 -> i32, i32, i32, i32
  %b2:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in2 : i32 -> i32, i32, i32, i32
  %b3:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in3 : i32 -> i32, i32, i32, i32
  %b4:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in4 : i32 -> i32, i32, i32, i32
  %b5:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in5 : i32 -> i32, i32, i32, i32
  %b6:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in6 : i32 -> i32, i32, i32, i32
  %b7:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in7 : i32 -> i32, i32, i32, i32
  %b8:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in8 : i32 -> i32, i32, i32, i32
  %b9:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in9 : i32 -> i32, i32, i32, i32
  %b10:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in10 : i32 -> i32, i32, i32, i32
  %b11:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in11 : i32 -> i32, i32, i32, i32
  %b12:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in12 : i32 -> i32, i32, i32, i32
  %b13:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in13 : i32 -> i32, i32, i32, i32
  %b14:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in14 : i32 -> i32, i32, i32, i32
  %b15:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in15 : i32 -> i32, i32, i32, i32

  // Row 0: input layer PEs
  %r0a = fabric.instance @pe_addi(%b0#0, %b1#0) {sym_name = "r0_addi_0"} : (i32, i32) -> i32
  %r0b = fabric.instance @pe_muli(%b2#0, %b3#0) {sym_name = "r0_muli_0"} : (i32, i32) -> i32
  %r0c = fabric.instance @pe_subi(%b4#0, %b5#0) {sym_name = "r0_subi_0"} : (i32, i32) -> i32
  %r0d = fabric.instance @pe_andi(%b6#0, %b7#0) {sym_name = "r0_andi_0"} : (i32, i32) -> i32
  %r0e = fabric.instance @pe_addi(%b8#0, %b9#0) {sym_name = "r0_addi_1"} : (i32, i32) -> i32
  %r0f = fabric.instance @pe_muli(%b10#0, %b11#0) {sym_name = "r0_muli_1"} : (i32, i32) -> i32
  %r0g = fabric.instance @pe_ori(%b12#0, %b13#0) {sym_name = "r0_ori_0"} : (i32, i32) -> i32
  %r0h = fabric.instance @pe_xori(%b14#0, %b15#0) {sym_name = "r0_xori_0"} : (i32, i32) -> i32
  %sw_r0a:4 = fabric.instance @sw4x4(%r0a, %r0b, %r0c, %r0d)
    {sym_name = "sw_r0a"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %sw_r0b:4 = fabric.instance @sw4x4(%r0e, %r0f, %r0g, %r0h)
    {sym_name = "sw_r0b"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Row 1: second layer PEs
  %r1a = fabric.instance @pe_addi(%b0#1, %b1#1) {sym_name = "r1_addi_0"} : (i32, i32) -> i32
  %r1b = fabric.instance @pe_muli(%b2#1, %b3#1) {sym_name = "r1_muli_0"} : (i32, i32) -> i32
  %r1c = fabric.instance @pe_shli(%b4#1, %b5#1) {sym_name = "r1_shli_0"} : (i32, i32) -> i32
  %r1d = fabric.instance @pe_shrui(%b6#1, %b7#1) {sym_name = "r1_shrui_0"} : (i32, i32) -> i32
  %r1e = fabric.instance @pe_addi(%b8#1, %b9#1) {sym_name = "r1_addi_1"} : (i32, i32) -> i32
  %r1f = fabric.instance @pe_subi(%b10#1, %b11#1) {sym_name = "r1_subi_0"} : (i32, i32) -> i32
  %r1g = fabric.instance @pe_shrsi(%b12#1, %b13#1) {sym_name = "r1_shrsi_0"} : (i32, i32) -> i32
  %r1h = fabric.instance @pe_divsi(%b14#1, %b15#1) {sym_name = "r1_divsi_0"} : (i32, i32) -> i32
  %sw_r1a:4 = fabric.instance @sw4x4(%r1a, %r1b, %r1c, %r1d)
    {sym_name = "sw_r1a"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %sw_r1b:4 = fabric.instance @sw4x4(%r1e, %r1f, %r1g, %r1h)
    {sym_name = "sw_r1b"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Row 2: cross-row connections
  %r2a = fabric.instance @pe_addi(%sw_r0a#0, %sw_r1a#0) {sym_name = "r2_addi_0"} : (i32, i32) -> i32
  %r2b = fabric.instance @pe_muli(%sw_r0a#1, %sw_r1a#1) {sym_name = "r2_muli_0"} : (i32, i32) -> i32
  %r2c = fabric.instance @pe_subi(%sw_r0a#2, %sw_r1a#2) {sym_name = "r2_subi_0"} : (i32, i32) -> i32
  %r2d = fabric.instance @pe_andi(%sw_r0a#3, %sw_r1a#3) {sym_name = "r2_andi_0"} : (i32, i32) -> i32
  %r2e = fabric.instance @pe_addi(%sw_r0b#0, %sw_r1b#0) {sym_name = "r2_addi_1"} : (i32, i32) -> i32
  %r2f = fabric.instance @pe_muli(%sw_r0b#1, %sw_r1b#1) {sym_name = "r2_muli_1"} : (i32, i32) -> i32
  %r2g = fabric.instance @pe_ori(%sw_r0b#2, %sw_r1b#2) {sym_name = "r2_ori_0"} : (i32, i32) -> i32
  %r2h = fabric.instance @pe_xori(%sw_r0b#3, %sw_r1b#3) {sym_name = "r2_xori_0"} : (i32, i32) -> i32
  %sw_r2a:4 = fabric.instance @sw4x4(%r2a, %r2b, %r2c, %r2d)
    {sym_name = "sw_r2a"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %sw_r2b:4 = fabric.instance @sw4x4(%r2e, %r2f, %r2g, %r2h)
    {sym_name = "sw_r2b"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Row 3: convergence layer
  %r3a = fabric.instance @pe_addi(%sw_r2a#0, %sw_r2b#0) {sym_name = "r3_addi_0"} : (i32, i32) -> i32
  %r3b = fabric.instance @pe_muli(%sw_r2a#1, %sw_r2b#1) {sym_name = "r3_muli_0"} : (i32, i32) -> i32
  %r3c = fabric.instance @pe_subi(%sw_r2a#2, %sw_r2b#2) {sym_name = "r3_subi_0"} : (i32, i32) -> i32
  %r3d = fabric.instance @pe_shli(%sw_r2a#3, %sw_r2b#3) {sym_name = "r3_shli_0"} : (i32, i32) -> i32
  %r3e = fabric.instance @pe_addi(%b0#2, %b1#2) {sym_name = "r3_addi_1"} : (i32, i32) -> i32
  %r3f = fabric.instance @pe_muli(%b2#2, %b3#2) {sym_name = "r3_muli_1"} : (i32, i32) -> i32
  %r3g = fabric.instance @pe_shrui(%b4#2, %b5#2) {sym_name = "r3_shrui_0"} : (i32, i32) -> i32
  %r3h = fabric.instance @pe_shrsi(%b6#2, %b7#2) {sym_name = "r3_shrsi_0"} : (i32, i32) -> i32
  %sw_r3a:4 = fabric.instance @sw4x4(%r3a, %r3b, %r3c, %r3d)
    {sym_name = "sw_r3a"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %sw_r3b:4 = fabric.instance @sw4x4(%r3e, %r3f, %r3g, %r3h)
    {sym_name = "sw_r3b"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Row 4-7: additional compute depth
  %r4a = fabric.instance @pe_addi(%sw_r3a#0, %sw_r3b#0) {sym_name = "r4_addi_0"} : (i32, i32) -> i32
  %r4b = fabric.instance @pe_muli(%sw_r3a#1, %sw_r3b#1) {sym_name = "r4_muli_0"} : (i32, i32) -> i32
  %r4c = fabric.instance @pe_subi(%sw_r3a#2, %sw_r3b#2) {sym_name = "r4_subi_0"} : (i32, i32) -> i32
  %r4d = fabric.instance @pe_andi(%sw_r3a#3, %sw_r3b#3) {sym_name = "r4_andi_0"} : (i32, i32) -> i32
  %r4e = fabric.instance @pe_addi(%b8#2, %b9#2) {sym_name = "r4_addi_1"} : (i32, i32) -> i32
  %r4f = fabric.instance @pe_muli(%b10#2, %b11#2) {sym_name = "r4_muli_1"} : (i32, i32) -> i32
  %r4g = fabric.instance @pe_ori(%b12#2, %b13#2) {sym_name = "r4_ori_0"} : (i32, i32) -> i32
  %r4h = fabric.instance @pe_xori(%b14#2, %b15#2) {sym_name = "r4_xori_0"} : (i32, i32) -> i32
  %sw_r4a:4 = fabric.instance @sw4x4(%r4a, %r4b, %r4c, %r4d)
    {sym_name = "sw_r4a"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %sw_r4b:4 = fabric.instance @sw4x4(%r4e, %r4f, %r4g, %r4h)
    {sym_name = "sw_r4b"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // Output layer
  %out0 = fabric.instance @pe_addi(%sw_r4a#0, %sw_r4b#0) {sym_name = "out_addi_0"} : (i32, i32) -> i32
  %out1 = fabric.instance @pe_muli(%sw_r4a#1, %sw_r4b#1) {sym_name = "out_muli_0"} : (i32, i32) -> i32
  %out2 = fabric.instance @pe_subi(%sw_r4a#2, %sw_r4b#2) {sym_name = "out_subi_0"} : (i32, i32) -> i32
  %out3 = fabric.instance @pe_addi(%sw_r4a#3, %sw_r4b#3) {sym_name = "out_addi_1"} : (i32, i32) -> i32
  %out4 = fabric.instance @pe_addi(%b0#3, %b1#3) {sym_name = "out_addi_2"} : (i32, i32) -> i32
  %out5 = fabric.instance @pe_muli(%b2#3, %b3#3) {sym_name = "out_muli_1"} : (i32, i32) -> i32
  %out6 = fabric.instance @pe_subi(%b4#3, %b5#3) {sym_name = "out_subi_1"} : (i32, i32) -> i32
  %out7 = fabric.instance @pe_andi(%b6#3, %b7#3) {sym_name = "out_andi_0"} : (i32, i32) -> i32

  // External memory modules
  %ld0, %done0 = fabric.extmemory
    [ldCount = 1, stCount = 0]
    (%mem0, %addr0)
    : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)
  %ld1, %done1 = fabric.extmemory
    [ldCount = 1, stCount = 0]
    (%mem1, %addr1)
    : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)
  %ld2, %done2 = fabric.extmemory
    [ldCount = 1, stCount = 0]
    (%mem2, %addr2)
    : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)
  %ld3, %done3 = fabric.extmemory
    [ldCount = 1, stCount = 0]
    (%mem3, %addr3)
    : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)

  fabric.yield %out0, %out1, %out2, %out3, %out4, %out5, %out6, %out7
    : i32, i32, i32, i32, i32, i32, i32, i32
}

}
