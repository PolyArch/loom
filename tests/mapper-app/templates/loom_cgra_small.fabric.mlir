// Small CGRA template (4x4 grid equivalent)
// Targets apps with <= 300 lines in handshake.mlir
//
// Resources: 4 compute tiles, 8 constants, 4 load + 4 store PEs,
//   2 private + 2 external memory, 4 dataflow PEs per type

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
// Compare and select PEs
// ---------------------------------------------------------------------------
fabric.pe @pe_cmpi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpi ult, %a, %b : i32
  fabric.yield %r : i1
}
fabric.pe @pe_select(%c: i1, %a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.select %c, %a, %b : i32
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
// Switch: 4-port full crossbar
// ---------------------------------------------------------------------------
fabric.switch @sw4x4 [connectivity_table = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

// ---------------------------------------------------------------------------
// FIFO: pipeline buffer
// ---------------------------------------------------------------------------
fabric.fifo @fifo_buf [depth = 2] : (i32) -> (i32)

// ---------------------------------------------------------------------------
// Top-level module
// ---------------------------------------------------------------------------
fabric.module @loom_cgra_small(
    %mem0: memref<?xi32>,
    %in0: i32, %in1: i32, %in2: i32, %in3: i32,
    %addr0: index
) -> (i32, i32) {
  // Broadcast inputs via switches (strict 1-to-1 fanout rule)
  %bin0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in0 : i32 -> i32, i32, i32, i32
  %bin1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in1 : i32 -> i32, i32, i32, i32
  %bin2:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in2 : i32 -> i32, i32, i32, i32
  %bin3:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %in3 : i32 -> i32, i32, i32, i32

  // Compute PEs
  %a0 = fabric.instance @pe_addi(%bin0#0, %bin1#0) {sym_name = "addi_0"} : (i32, i32) -> i32
  %ba0:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a0 : i32 -> i32, i32, i32, i32
  %a1 = fabric.instance @pe_addi(%bin2#0, %bin3#0) {sym_name = "addi_1"} : (i32, i32) -> i32
  %ba1:4 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a1 : i32 -> i32, i32, i32, i32
  %s0 = fabric.instance @pe_subi(%ba0#0, %ba1#0) {sym_name = "subi_0"} : (i32, i32) -> i32
  %m0 = fabric.instance @pe_muli(%ba0#1, %ba1#1) {sym_name = "muli_0"} : (i32, i32) -> i32

  // Routing crossbar
  %sw0:4 = fabric.instance @sw4x4(%ba0#2, %ba1#2, %s0, %m0)
    {sym_name = "sw_0"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)
  %sw1:4 = fabric.instance @sw4x4(%sw0#0, %sw0#1, %sw0#2, %sw0#3)
    {sym_name = "sw_1"} : (i32, i32, i32, i32) -> (i32, i32, i32, i32)

  // External memory
  %ld0, %done0 = fabric.extmemory
    [ldCount = 1, stCount = 0]
    (%mem0, %addr0)
    : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)

  fabric.yield %sw1#0, %sw1#1 : i32, i32
}

}
