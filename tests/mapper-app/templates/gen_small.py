#!/usr/bin/env python3
"""Generate small CGRA fabric template with comprehensive PE coverage.

Produces loom_cgra_small.fabric.mlir with:
- Per-type-plane central routing switches with full connectivity
- Complete PE catalog covering all operations used by quick-tier apps
- Adequate PE instance counts to support routing of typical DFGs

Usage:
    python3 gen_small.py > loom_cgra_small.fabric.mlir
"""

import sys
from dataclasses import dataclass, field


def full_ct(n_in, n_out):
    """Full connectivity table (all-ones) for n_in x n_out switch."""
    return ", ".join(["1"] * (n_in * n_out))


# ---------------------------------------------------------------------------
# PE definition templates
# ---------------------------------------------------------------------------

PE_DEFS = """\
// Constant PEs
fabric.pe @pe_const_i32(%ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = handshake.constant %ctrl {value = 0 : i32} : i32
  fabric.yield %r : i32
}
fabric.pe @pe_const_f32(%ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = handshake.constant %ctrl {value = 0.0 : f32} : f32
  fabric.yield %r : f32
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

// Control flow PEs
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
fabric.pe @pe_join5(%a: none, %b: none, %c: none, %d: none, %e: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.join %a, %b, %c, %d, %e : none, none, none, none, none
  fabric.yield %r : none
}
fabric.pe @pe_join6(%a: none, %b: none, %c: none, %d: none, %e: none, %f: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.join %a, %b, %c, %d, %e, %f : none, none, none, none, none, none
  fabric.yield %r : none
}
fabric.pe @pe_join_i1(%a: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.join %a : i1
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
fabric.pe @pe_cond_br_f32(%cond: i1, %data: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32, f32) {
  %t, %f = handshake.cond_br %cond, %data : f32
  fabric.yield %t, %f : f32, f32
}
fabric.pe @pe_cond_br_index(%cond: i1, %data: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index, index) {
  %t, %f = handshake.cond_br %cond, %data : index
  fabric.yield %t, %f : index, index
}
fabric.pe @pe_mux_i32(%sel: index, %a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = handshake.mux %sel [%a, %b] : index, i32
  fabric.yield %r : i32
}
fabric.pe @pe_mux_f32(%sel: index, %a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = handshake.mux %sel [%a, %b] : index, f32
  fabric.yield %r : f32
}
fabric.pe @pe_mux_none(%sel: index, %a: none, %b: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %r = handshake.mux %sel [%a, %b] : index, none
  fabric.yield %r : none
}
fabric.pe @pe_mux_index(%sel: index, %a: index, %b: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = handshake.mux %sel [%a, %b] : index, index
  fabric.yield %r : index
}
fabric.pe @pe_mux_i64(%sel: index, %a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = handshake.mux %sel [%a, %b] : index, i64
  fabric.yield %r : i64
}

// Integer arithmetic PEs (i32)
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
fabric.pe @pe_divui(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.divui %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_divsi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.divsi %a, %b : i32
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
fabric.pe @pe_remui(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.remui %a, %b : i32
  fabric.yield %r : i32
}

// Index arithmetic PEs
fabric.pe @pe_addi_index(%a: index, %b: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.addi %a, %b : index
  fabric.yield %r : index
}
fabric.pe @pe_subi_index(%a: index, %b: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.subi %a, %b : index
  fabric.yield %r : index
}
fabric.pe @pe_muli_index(%a: index, %b: index) [latency = [1 : i16, 1 : i16, 2 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.muli %a, %b : index
  fabric.yield %r : index
}
fabric.pe @pe_divui_index(%a: index, %b: index) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.divui %a, %b : index
  fabric.yield %r : index
}
fabric.pe @pe_divsi_index(%a: index, %b: index) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.divsi %a, %b : index
  fabric.yield %r : index
}
fabric.pe @pe_remui_index(%a: index, %b: index) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.remui %a, %b : index
  fabric.yield %r : index
}

// i64 arithmetic PEs
fabric.pe @pe_addi_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.addi %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_subi_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.subi %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_muli_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 2 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.muli %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_andi_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.andi %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_ori_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.ori %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_xori_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.xori %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_shli_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.shli %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_shrui_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.shrui %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_shrsi_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.shrsi %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_remui_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.remui %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_divui_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.divui %a, %b : i64
  fabric.yield %r : i64
}
fabric.pe @pe_divsi_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.divsi %a, %b : i64
  fabric.yield %r : i64
}

// Floating point arithmetic PEs
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
fabric.pe @pe_absf(%a: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = math.absf %a : f32
  fabric.yield %r : f32
}
fabric.pe @pe_fma(%a: f32, %b: f32, %c: f32) [latency = [1 : i16, 1 : i16, 3 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = math.fma %a, %b, %c : f32
  fabric.yield %r : f32
}
fabric.pe @pe_sinf(%a: f32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = math.sin %a : f32
  fabric.yield %r : f32
}
fabric.pe @pe_cosf(%a: f32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = math.cos %a : f32
  fabric.yield %r : f32
}
fabric.pe @pe_expf(%a: f32) [latency = [1 : i16, 1 : i16, 10 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = math.exp %a : f32
  fabric.yield %r : f32
}

// Compare PEs
fabric.pe @pe_cmpi(%a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpi ult, %a, %b : i32
  fabric.yield %r : i1
}
fabric.pe @pe_cmpi_i64(%a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpi ult, %a, %b : i64
  fabric.yield %r : i1
}
fabric.pe @pe_cmpi_index(%a: index, %b: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpi ult, %a, %b : index
  fabric.yield %r : i1
}
fabric.pe @pe_cmpf(%a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %r = arith.cmpf ult, %a, %b : f32
  fabric.yield %r : i1
}

// Select PEs
fabric.pe @pe_select(%c: i1, %a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.select %c, %a, %b : i32
  fabric.yield %r : i32
}
fabric.pe @pe_select_index(%c: i1, %a: index, %b: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.select %c, %a, %b : index
  fabric.yield %r : index
}
fabric.pe @pe_select_f32(%c: i1, %a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.select %c, %a, %b : f32
  fabric.yield %r : f32
}
fabric.pe @pe_select_i64(%c: i1, %a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.select %c, %a, %b : i64
  fabric.yield %r : i64
}

// Type conversion PEs
fabric.pe @pe_index_cast_i64(%v: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.index_cast %v : i64 to index
  fabric.yield %r : index
}
fabric.pe @pe_index_cast_i32(%v: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.index_cast %v : i32 to index
  fabric.yield %r : index
}
fabric.pe @pe_index_castui(%v: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %r = arith.index_castui %v : i32 to index
  fabric.yield %r : index
}
fabric.pe @pe_index_cast_to_i64(%v: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.index_cast %v : index to i64
  fabric.yield %r : i64
}
fabric.pe @pe_index_cast_to_i32(%v: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.index_cast %v : index to i32
  fabric.yield %r : i32
}
fabric.pe @pe_extui(%a: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.extui %a : i32 to i64
  fabric.yield %r : i64
}
fabric.pe @pe_extsi(%a: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %r = arith.extsi %a : i32 to i64
  fabric.yield %r : i64
}
fabric.pe @pe_trunci(%a: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.trunci %a : i64 to i32
  fabric.yield %r : i32
}
fabric.pe @pe_uitofp(%a: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.uitofp %a : i32 to f32
  fabric.yield %r : f32
}
fabric.pe @pe_sitofp(%a: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %r = arith.sitofp %a : i32 to f32
  fabric.yield %r : f32
}
fabric.pe @pe_fptoui(%a: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.fptoui %a : f32 to i32
  fabric.yield %r : i32
}
fabric.pe @pe_fptosi(%a: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %r = arith.fptosi %a : f32 to i32
  fabric.yield %r : i32
}

// Load and store PEs
fabric.pe @pe_load(%addr: index, %data: i32, %ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, index) {
  %ld_d, %ld_a = handshake.load [%addr] %data, %ctrl : index, i32
  fabric.yield %ld_d, %ld_a : i32, index
}
fabric.pe @pe_load_f32(%addr: index, %data: f32, %ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32, index) {
  %ld_d, %ld_a = handshake.load [%addr] %data, %ctrl : index, f32
  fabric.yield %ld_d, %ld_a : f32, index
}
fabric.pe @pe_store(%addr: index, %data: i32, %ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, index) {
  %st_d, %st_a = handshake.store [%addr] %data, %ctrl : index, i32
  fabric.yield %st_d, %st_a : i32, index
}
fabric.pe @pe_store_f32(%addr: index, %data: f32, %ctrl: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32, index) {
  %st_d, %st_a = handshake.store [%addr] %data, %ctrl : index, f32
  fabric.yield %st_d, %st_a : f32, index
}

// Dataflow PEs
fabric.pe @pe_invariant(%d: i1, %a: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %o = dataflow.invariant %d, %a : i1, i32 -> i32
  fabric.yield %o : i32
}
fabric.pe @pe_invariant_i1(%d: i1, %a: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i1) {
  %o = dataflow.invariant %d, %a : i1, i1 -> i1
  fabric.yield %o : i1
}
fabric.pe @pe_invariant_none(%d: i1, %a: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %o = dataflow.invariant %d, %a : i1, none -> none
  fabric.yield %o : none
}
fabric.pe @pe_invariant_index(%d: i1, %a: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %o = dataflow.invariant %d, %a : i1, index -> index
  fabric.yield %o : index
}
fabric.pe @pe_invariant_f32(%d: i1, %a: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %o = dataflow.invariant %d, %a : i1, f32 -> f32
  fabric.yield %o : f32
}
fabric.pe @pe_invariant_i64(%d: i1, %a: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %o = dataflow.invariant %d, %a : i1, i64 -> i64
  fabric.yield %o : i64
}
fabric.pe @pe_carry(%d: i1, %a: i32, %b: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32) {
  %o = dataflow.carry %d, %a, %b : i1, i32, i32 -> i32
  fabric.yield %o : i32
}
fabric.pe @pe_carry_f32(%d: i1, %a: f32, %b: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32) {
  %o = dataflow.carry %d, %a, %b : i1, f32, f32 -> f32
  fabric.yield %o : f32
}
fabric.pe @pe_carry_none(%d: i1, %a: none, %b: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  %o = dataflow.carry %d, %a, %b : i1, none, none -> none
  fabric.yield %o : none
}
fabric.pe @pe_carry_index(%d: i1, %a: index, %b: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (index) {
  %o = dataflow.carry %d, %a, %b : i1, index, index -> index
  fabric.yield %o : index
}
fabric.pe @pe_carry_i64(%d: i1, %a: i64, %b: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i64) {
  %o = dataflow.carry %d, %a, %b : i1, i64, i64 -> i64
  fabric.yield %o : i64
}
fabric.pe @pe_gate(%val: i32, %cond: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (i32, i1) {
  %av, %ac = dataflow.gate %val, %cond : i32, i1 -> i32, i1
  fabric.yield %av, %ac : i32, i1
}
fabric.pe @pe_gate_f32(%val: f32, %cond: i1) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (f32, i1) {
  %av, %ac = dataflow.gate %val, %cond : f32, i1 -> f32, i1
  fabric.yield %av, %ac : f32, i1
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
fabric.pe @pe_sink_i32(%v: i32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> () {
  handshake.sink %v : i32
  fabric.yield
}
fabric.pe @pe_sink_index(%v: index) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> () {
  handshake.sink %v : index
  fabric.yield
}
fabric.pe @pe_sink_f32(%v: f32) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> () {
  handshake.sink %v : f32
  fabric.yield
}
fabric.pe @pe_sink_i64(%v: i64) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> () {
  handshake.sink %v : i64
  fabric.yield
}
"""

# ---------------------------------------------------------------------------
# Port tracking for switch sizing
# ---------------------------------------------------------------------------

@dataclass
class SwitchPlane:
    """Tracks inputs and outputs for a single type plane's central switch."""
    typ: str
    inputs: list = field(default_factory=list)    # SSA values feeding IN
    outputs: list = field(default_factory=list)   # SSA values consumed OUT

    @property
    def n_in(self):
        return len(self.inputs)

    @property
    def n_out(self):
        return len(self.outputs)


def generate_module(out):
    """Generate the fabric.module body with proper port tracking."""

    # Track switch planes
    planes = {
        "i32": SwitchPlane("i32"),
        "f32": SwitchPlane("f32"),
        "i64": SwitchPlane("i64"),
        "index": SwitchPlane("index"),
        "i1": SwitchPlane("i1"),
        "none": SwitchPlane("none"),
    }

    # Collect all instance lines and track ports
    instances = []  # (comment, lines)
    extmem_lines = []

    def sw_in(typ, name):
        """Register an input to a switch plane, return the SSA name."""
        planes[typ].inputs.append(name)

    def sw_out(typ):
        """Allocate an output from a switch plane, return the port ref."""
        idx = planes[typ].n_out
        planes[typ].outputs.append(idx)
        return f"%csw_{typ.replace('i', 'i').replace('index', 'idx').replace('none', 'n').replace('f32', 'f32')}#{idx}"

    # Simplified: use csw_XXX naming
    def port(typ):
        """Get next output port from type switch."""
        tname = {"i32": "i32", "f32": "f32", "i64": "i64",
                 "index": "idx", "i1": "i1", "none": "n"}[typ]
        idx = planes[typ].n_out
        planes[typ].outputs.append(idx)
        return f"%csw_{tname}#{idx}"

    def feed(typ, name):
        """Register an SSA value as input to the type switch."""
        planes[typ].inputs.append(name)

    lines = []

    # --- Constant PEs ---
    lines.append("  // Constant PEs")
    for i in range(12):
        lines.append(f"  %ci32_{i} = fabric.instance @pe_const_i32({port('none')}) "
                     f'{{sym_name = "ci32_{i}"}} : (none) -> i32')
        feed("i32", f"%ci32_{i}")
    for i in range(6):
        lines.append(f"  %cf32_{i} = fabric.instance @pe_const_f32({port('none')}) "
                     f'{{sym_name = "cf32_{i}"}} : (none) -> f32')
        feed("f32", f"%cf32_{i}")
    for i in range(4):
        lines.append(f"  %ci64_{i} = fabric.instance @pe_const_i64({port('none')}) "
                     f'{{sym_name = "ci64_{i}"}} : (none) -> i64')
        feed("i64", f"%ci64_{i}")
    for i in range(8):
        lines.append(f"  %cidx_{i} = fabric.instance @pe_const_index({port('none')}) "
                     f'{{sym_name = "cidx_{i}"}} : (none) -> index')
        feed("index", f"%cidx_{i}")
    for i in range(4):
        lines.append(f"  %ci1_{i} = fabric.instance @pe_const_i1({port('none')}) "
                     f'{{sym_name = "ci1_{i}"}} : (none) -> i1')
        feed("i1", f"%ci1_{i}")

    # --- Join PEs ---
    lines.append("  // Join PEs")
    for i in range(6):
        lines.append(f"  %join_{i} = fabric.instance @pe_join1({port('none')}) "
                     f'{{sym_name = "join_{i}"}} : (none) -> none')
        feed("none", f"%join_{i}")
    for i in range(3):
        lines.append(f"  %join2_{i} = fabric.instance @pe_join2({port('none')}, {port('none')}) "
                     f'{{sym_name = "join2_{i}"}} : (none, none) -> none')
        feed("none", f"%join2_{i}")
    for i in range(2):
        lines.append(f"  %join3_{i} = fabric.instance @pe_join3({port('none')}, {port('none')}, {port('none')}) "
                     f'{{sym_name = "join3_{i}"}} : (none, none, none) -> none')
        feed("none", f"%join3_{i}")
    for i in range(2):
        lines.append(f"  %join4_{i} = fabric.instance @pe_join4({port('none')}, {port('none')}, {port('none')}, {port('none')}) "
                     f'{{sym_name = "join4_{i}"}} : (none, none, none, none) -> none')
        feed("none", f"%join4_{i}")
    lines.append(f"  %join5_0 = fabric.instance @pe_join5({port('none')}, {port('none')}, {port('none')}, {port('none')}, {port('none')}) "
                 f'{{sym_name = "join5_0"}} : (none, none, none, none, none) -> none')
    feed("none", "%join5_0")
    lines.append(f"  %join6_0 = fabric.instance @pe_join6({port('none')}, {port('none')}, {port('none')}, {port('none')}, {port('none')}, {port('none')}) "
                 f'{{sym_name = "join6_0"}} : (none, none, none, none, none, none) -> none')
    feed("none", "%join6_0")
    lines.append(f"  %join_i1_0 = fabric.instance @pe_join_i1({port('i1')}) "
                 f'{{sym_name = "join_i1_0"}} : (i1) -> none')
    feed("none", "%join_i1_0")

    # --- Compare PEs ---
    lines.append("  // Compare PEs")
    for i in range(5):
        lines.append(f"  %cmpi_{i} = fabric.instance @pe_cmpi({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "cmpi_{i}"}} : (i32, i32) -> i1')
        feed("i1", f"%cmpi_{i}")
    for i in range(6):
        lines.append(f"  %cmpi_i64_{i} = fabric.instance @pe_cmpi_i64({port('i64')}, {port('i64')}) "
                     f'{{sym_name = "cmpi_i64_{i}"}} : (i64, i64) -> i1')
        feed("i1", f"%cmpi_i64_{i}")
    for i in range(4):
        lines.append(f"  %cmpi_idx_{i} = fabric.instance @pe_cmpi_index({port('index')}, {port('index')}) "
                     f'{{sym_name = "cmpi_idx_{i}"}} : (index, index) -> i1')
        feed("i1", f"%cmpi_idx_{i}")
    for i in range(2):
        lines.append(f"  %cmpf_{i} = fabric.instance @pe_cmpf({port('f32')}, {port('f32')}) "
                     f'{{sym_name = "cmpf_{i}"}} : (f32, f32) -> i1')
        feed("i1", f"%cmpf_{i}")

    # --- Type conversion PEs ---
    lines.append("  // Type conversion PEs")
    for i in range(6):
        lines.append(f"  %icast64_{i} = fabric.instance @pe_index_cast_i64({port('i64')}) "
                     f'{{sym_name = "icast64_{i}"}} : (i64) -> index')
        feed("index", f"%icast64_{i}")
    for i in range(6):
        lines.append(f"  %icast32_{i} = fabric.instance @pe_index_cast_i32({port('i32')}) "
                     f'{{sym_name = "icast32_{i}"}} : (i32) -> index')
        feed("index", f"%icast32_{i}")
    for i in range(2):
        lines.append(f"  %icastui_{i} = fabric.instance @pe_index_castui({port('i32')}) "
                     f'{{sym_name = "icastui_{i}"}} : (i32) -> index')
        feed("index", f"%icastui_{i}")
    for i in range(3):
        lines.append(f"  %extui_{i} = fabric.instance @pe_extui({port('i32')}) "
                     f'{{sym_name = "extui_{i}"}} : (i32) -> i64')
        feed("i64", f"%extui_{i}")
    for i in range(2):
        lines.append(f"  %extsi_{i} = fabric.instance @pe_extsi({port('i32')}) "
                     f'{{sym_name = "extsi_{i}"}} : (i32) -> i64')
        feed("i64", f"%extsi_{i}")
    for i in range(2):
        lines.append(f"  %trunci_{i} = fabric.instance @pe_trunci({port('i64')}) "
                     f'{{sym_name = "trunci_{i}"}} : (i64) -> i32')
        feed("i32", f"%trunci_{i}")
    for i in range(5):
        lines.append(f"  %uitofp_{i} = fabric.instance @pe_uitofp({port('i32')}) "
                     f'{{sym_name = "uitofp_{i}"}} : (i32) -> f32')
        feed("f32", f"%uitofp_{i}")
    for i in range(2):
        lines.append(f"  %sitofp_{i} = fabric.instance @pe_sitofp({port('i32')}) "
                     f'{{sym_name = "sitofp_{i}"}} : (i32) -> f32')
        feed("f32", f"%sitofp_{i}")
    for i in range(2):
        lines.append(f"  %fptoui_{i} = fabric.instance @pe_fptoui({port('f32')}) "
                     f'{{sym_name = "fptoui_{i}"}} : (f32) -> i32')
        feed("i32", f"%fptoui_{i}")
    for i in range(2):
        lines.append(f"  %fptosi_{i} = fabric.instance @pe_fptosi({port('f32')}) "
                     f'{{sym_name = "fptosi_{i}"}} : (f32) -> i32')
        feed("i32", f"%fptosi_{i}")
    for i in range(2):
        lines.append(f"  %icast_to_i32_{i} = fabric.instance @pe_index_cast_to_i32({port('index')}) "
                     f'{{sym_name = "icast_to_i32_{i}"}} : (index) -> i32')
        feed("i32", f"%icast_to_i32_{i}")
    for i in range(2):
        lines.append(f"  %icast_to_i64_{i} = fabric.instance @pe_index_cast_to_i64({port('index')}) "
                     f'{{sym_name = "icast_to_i64_{i}"}} : (index) -> i64')
        feed("i64", f"%icast_to_i64_{i}")

    # --- Select PEs ---
    lines.append("  // Select PEs")
    for i in range(3):
        lines.append(f"  %sel_{i} = fabric.instance @pe_select({port('i1')}, {port('i32')}, {port('i32')}) "
                     f'{{sym_name = "sel_{i}"}} : (i1, i32, i32) -> i32')
        feed("i32", f"%sel_{i}")
    for i in range(3):
        lines.append(f"  %sel_idx_{i} = fabric.instance @pe_select_index({port('i1')}, {port('index')}, {port('index')}) "
                     f'{{sym_name = "sel_idx_{i}"}} : (i1, index, index) -> index')
        feed("index", f"%sel_idx_{i}")
    for i in range(2):
        lines.append(f"  %sel_f_{i} = fabric.instance @pe_select_f32({port('i1')}, {port('f32')}, {port('f32')}) "
                     f'{{sym_name = "sel_f_{i}"}} : (i1, f32, f32) -> f32')
        feed("f32", f"%sel_f_{i}")
    for i in range(2):
        lines.append(f"  %sel_i64_{i} = fabric.instance @pe_select_i64({port('i1')}, {port('i64')}, {port('i64')}) "
                     f'{{sym_name = "sel_i64_{i}"}} : (i1, i64, i64) -> i64')
        feed("i64", f"%sel_i64_{i}")

    # --- Dataflow PEs ---
    lines.append("  // Dataflow PEs")
    for i in range(3):
        lines.append(f"  %stream_{i}:2 = fabric.instance @pe_stream({port('index')}, {port('index')}, {port('index')}) "
                     f'{{sym_name = "stream_{i}"}} : (index, index, index) -> (index, i1)')
        feed("index", f"%stream_{i}#0")
        feed("i1", f"%stream_{i}#1")

    for i in range(3):
        lines.append(f"  %gate_{i}:2 = fabric.instance @pe_gate({port('i32')}, {port('i1')}) "
                     f'{{sym_name = "gate_{i}"}} : (i32, i1) -> (i32, i1)')
        feed("i32", f"%gate_{i}#0")
        feed("i1", f"%gate_{i}#1")
    for i in range(2):
        lines.append(f"  %gate_f32_{i}:2 = fabric.instance @pe_gate_f32({port('f32')}, {port('i1')}) "
                     f'{{sym_name = "gate_f32_{i}"}} : (f32, i1) -> (f32, i1)')
        feed("f32", f"%gate_f32_{i}#0")
        feed("i1", f"%gate_f32_{i}#1")
    for i in range(2):
        lines.append(f"  %gate_idx_{i}:2 = fabric.instance @pe_gate_index({port('index')}, {port('i1')}) "
                     f'{{sym_name = "gate_idx_{i}"}} : (index, i1) -> (index, i1)')
        feed("index", f"%gate_idx_{i}#0")
        feed("i1", f"%gate_idx_{i}#1")

    for i in range(4):
        lines.append(f"  %carry_{i} = fabric.instance @pe_carry({port('i1')}, {port('i32')}, {port('i32')}) "
                     f'{{sym_name = "carry_{i}"}} : (i1, i32, i32) -> i32')
        feed("i32", f"%carry_{i}")
    for i in range(3):
        lines.append(f"  %carry_f32_{i} = fabric.instance @pe_carry_f32({port('i1')}, {port('f32')}, {port('f32')}) "
                     f'{{sym_name = "carry_f32_{i}"}} : (i1, f32, f32) -> f32')
        feed("f32", f"%carry_f32_{i}")
    for i in range(6):
        lines.append(f"  %carry_n_{i} = fabric.instance @pe_carry_none({port('i1')}, {port('none')}, {port('none')}) "
                     f'{{sym_name = "carry_n_{i}"}} : (i1, none, none) -> none')
        feed("none", f"%carry_n_{i}")
    for i in range(2):
        lines.append(f"  %carry_idx_{i} = fabric.instance @pe_carry_index({port('i1')}, {port('index')}, {port('index')}) "
                     f'{{sym_name = "carry_idx_{i}"}} : (i1, index, index) -> index')
        feed("index", f"%carry_idx_{i}")
    for i in range(2):
        lines.append(f"  %carry_i64_{i} = fabric.instance @pe_carry_i64({port('i1')}, {port('i64')}, {port('i64')}) "
                     f'{{sym_name = "carry_i64_{i}"}} : (i1, i64, i64) -> i64')
        feed("i64", f"%carry_i64_{i}")

    for i in range(2):
        lines.append(f"  %inv_{i} = fabric.instance @pe_invariant({port('i1')}, {port('i32')}) "
                     f'{{sym_name = "inv_{i}"}} : (i1, i32) -> i32')
        feed("i32", f"%inv_{i}")
    for i in range(3):
        lines.append(f"  %inv_f32_{i} = fabric.instance @pe_invariant_f32({port('i1')}, {port('f32')}) "
                     f'{{sym_name = "inv_f32_{i}"}} : (i1, f32) -> f32')
        feed("f32", f"%inv_f32_{i}")
    for i in range(2):
        lines.append(f"  %inv_idx_{i} = fabric.instance @pe_invariant_index({port('i1')}, {port('index')}) "
                     f'{{sym_name = "inv_idx_{i}"}} : (i1, index) -> index')
        feed("index", f"%inv_idx_{i}")
    for i in range(2):
        lines.append(f"  %inv_i1_{i} = fabric.instance @pe_invariant_i1({port('i1')}, {port('i1')}) "
                     f'{{sym_name = "inv_i1_{i}"}} : (i1, i1) -> i1')
        feed("i1", f"%inv_i1_{i}")
    for i in range(2):
        lines.append(f"  %inv_n_{i} = fabric.instance @pe_invariant_none({port('i1')}, {port('none')}) "
                     f'{{sym_name = "inv_n_{i}"}} : (i1, none) -> none')
        feed("none", f"%inv_n_{i}")
    for i in range(2):
        lines.append(f"  %inv_i64_{i} = fabric.instance @pe_invariant_i64({port('i1')}, {port('i64')}) "
                     f'{{sym_name = "inv_i64_{i}"}} : (i1, i64) -> i64')
        feed("i64", f"%inv_i64_{i}")

    # --- i32 arithmetic PEs ---
    lines.append("  // i32 arithmetic PEs")
    for i in range(5):
        lines.append(f"  %addi_{i} = fabric.instance @pe_addi({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "addi_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%addi_{i}")
    for i in range(2):
        lines.append(f"  %subi_{i} = fabric.instance @pe_subi({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "subi_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%subi_{i}")
    for i in range(3):
        lines.append(f"  %muli_{i} = fabric.instance @pe_muli({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "muli_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%muli_{i}")
    for i in range(2):
        lines.append(f"  %divui_{i} = fabric.instance @pe_divui({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "divui_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%divui_{i}")
    for i in range(2):
        lines.append(f"  %divsi_{i} = fabric.instance @pe_divsi({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "divsi_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%divsi_{i}")
    for i in range(2):
        lines.append(f"  %andi_{i} = fabric.instance @pe_andi({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "andi_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%andi_{i}")
    for i in range(2):
        lines.append(f"  %ori_{i} = fabric.instance @pe_ori({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "ori_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%ori_{i}")
    for i in range(3):
        lines.append(f"  %xori_{i} = fabric.instance @pe_xori({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "xori_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%xori_{i}")
    for i in range(3):
        lines.append(f"  %shli_{i} = fabric.instance @pe_shli({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "shli_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%shli_{i}")
    for i in range(2):
        lines.append(f"  %shrui_{i} = fabric.instance @pe_shrui({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "shrui_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%shrui_{i}")
    for i in range(2):
        lines.append(f"  %shrsi_{i} = fabric.instance @pe_shrsi({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "shrsi_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%shrsi_{i}")
    for i in range(2):
        lines.append(f"  %remui_{i} = fabric.instance @pe_remui({port('i32')}, {port('i32')}) "
                     f'{{sym_name = "remui_{i}"}} : (i32, i32) -> i32')
        feed("i32", f"%remui_{i}")

    # --- index arithmetic PEs ---
    lines.append("  // index arithmetic PEs")
    lines.append(f"  %addi_idx_0 = fabric.instance @pe_addi_index({port('index')}, {port('index')}) "
                 f'{{sym_name = "addi_idx_0"}} : (index, index) -> index')
    feed("index", "%addi_idx_0")
    lines.append(f"  %subi_idx_0 = fabric.instance @pe_subi_index({port('index')}, {port('index')}) "
                 f'{{sym_name = "subi_idx_0"}} : (index, index) -> index')
    feed("index", "%subi_idx_0")
    lines.append(f"  %muli_idx_0 = fabric.instance @pe_muli_index({port('index')}, {port('index')}) "
                 f'{{sym_name = "muli_idx_0"}} : (index, index) -> index')
    feed("index", "%muli_idx_0")
    lines.append(f"  %divui_idx_0 = fabric.instance @pe_divui_index({port('index')}, {port('index')}) "
                 f'{{sym_name = "divui_idx_0"}} : (index, index) -> index')
    feed("index", "%divui_idx_0")
    lines.append(f"  %divsi_idx_0 = fabric.instance @pe_divsi_index({port('index')}, {port('index')}) "
                 f'{{sym_name = "divsi_idx_0"}} : (index, index) -> index')
    feed("index", "%divsi_idx_0")
    lines.append(f"  %remui_idx_0 = fabric.instance @pe_remui_index({port('index')}, {port('index')}) "
                 f'{{sym_name = "remui_idx_0"}} : (index, index) -> index')
    feed("index", "%remui_idx_0")

    # --- i64 arithmetic PEs ---
    lines.append("  // i64 arithmetic PEs")
    lines.append(f"  %addi_i64_0 = fabric.instance @pe_addi_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "addi_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%addi_i64_0")
    lines.append(f"  %subi_i64_0 = fabric.instance @pe_subi_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "subi_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%subi_i64_0")
    lines.append(f"  %muli_i64_0 = fabric.instance @pe_muli_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "muli_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%muli_i64_0")
    lines.append(f"  %andi_i64_0 = fabric.instance @pe_andi_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "andi_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%andi_i64_0")
    lines.append(f"  %xori_i64_0 = fabric.instance @pe_xori_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "xori_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%xori_i64_0")
    lines.append(f"  %shli_i64_0 = fabric.instance @pe_shli_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "shli_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%shli_i64_0")
    lines.append(f"  %shrui_i64_0 = fabric.instance @pe_shrui_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "shrui_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%shrui_i64_0")
    lines.append(f"  %shrsi_i64_0 = fabric.instance @pe_shrsi_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "shrsi_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%shrsi_i64_0")
    lines.append(f"  %remui_i64_0 = fabric.instance @pe_remui_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "remui_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%remui_i64_0")
    lines.append(f"  %divui_i64_0 = fabric.instance @pe_divui_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "divui_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%divui_i64_0")
    lines.append(f"  %divsi_i64_0 = fabric.instance @pe_divsi_i64({port('i64')}, {port('i64')}) "
                 f'{{sym_name = "divsi_i64_0"}} : (i64, i64) -> i64')
    feed("i64", "%divsi_i64_0")

    # --- f32 arithmetic PEs ---
    lines.append("  // f32 arithmetic PEs")
    for i in range(3):
        lines.append(f"  %addf_{i} = fabric.instance @pe_addf({port('f32')}, {port('f32')}) "
                     f'{{sym_name = "addf_{i}"}} : (f32, f32) -> f32')
        feed("f32", f"%addf_{i}")
    for i in range(2):
        lines.append(f"  %subf_{i} = fabric.instance @pe_subf({port('f32')}, {port('f32')}) "
                     f'{{sym_name = "subf_{i}"}} : (f32, f32) -> f32')
        feed("f32", f"%subf_{i}")
    for i in range(3):
        lines.append(f"  %mulf_{i} = fabric.instance @pe_mulf({port('f32')}, {port('f32')}) "
                     f'{{sym_name = "mulf_{i}"}} : (f32, f32) -> f32')
        feed("f32", f"%mulf_{i}")
    for i in range(3):
        lines.append(f"  %divf_{i} = fabric.instance @pe_divf({port('f32')}, {port('f32')}) "
                     f'{{sym_name = "divf_{i}"}} : (f32, f32) -> f32')
        feed("f32", f"%divf_{i}")
    lines.append(f"  %negf_0 = fabric.instance @pe_negf({port('f32')}) "
                 f'{{sym_name = "negf_0"}} : (f32) -> f32')
    feed("f32", "%negf_0")
    lines.append(f"  %absf_0 = fabric.instance @pe_absf({port('f32')}) "
                 f'{{sym_name = "absf_0"}} : (f32) -> f32')
    feed("f32", "%absf_0")
    lines.append(f"  %fma_0 = fabric.instance @pe_fma({port('f32')}, {port('f32')}, {port('f32')}) "
                 f'{{sym_name = "fma_0"}} : (f32, f32, f32) -> f32')
    feed("f32", "%fma_0")
    lines.append(f"  %sinf_0 = fabric.instance @pe_sinf({port('f32')}) "
                 f'{{sym_name = "sinf_0"}} : (f32) -> f32')
    feed("f32", "%sinf_0")
    lines.append(f"  %cosf_0 = fabric.instance @pe_cosf({port('f32')}) "
                 f'{{sym_name = "cosf_0"}} : (f32) -> f32')
    feed("f32", "%cosf_0")
    lines.append(f"  %expf_0 = fabric.instance @pe_expf({port('f32')}) "
                 f'{{sym_name = "expf_0"}} : (f32) -> f32')
    feed("f32", "%expf_0")

    # --- Cond_br PEs ---
    lines.append("  // Conditional branch PEs")
    for i in range(8):
        lines.append(f"  %cbr_n_{i}:2 = fabric.instance @pe_cond_br_none({port('i1')}, {port('none')}) "
                     f'{{sym_name = "cbr_n_{i}"}} : (i1, none) -> (none, none)')
        feed("none", f"%cbr_n_{i}#0")
        feed("none", f"%cbr_n_{i}#1")
    for i in range(4):
        lines.append(f"  %cbr_i_{i}:2 = fabric.instance @pe_cond_br_i32({port('i1')}, {port('i32')}) "
                     f'{{sym_name = "cbr_i_{i}"}} : (i1, i32) -> (i32, i32)')
        feed("i32", f"%cbr_i_{i}#0")
        feed("i32", f"%cbr_i_{i}#1")
    for i in range(2):
        lines.append(f"  %cbr_f_{i}:2 = fabric.instance @pe_cond_br_f32({port('i1')}, {port('f32')}) "
                     f'{{sym_name = "cbr_f_{i}"}} : (i1, f32) -> (f32, f32)')
        feed("f32", f"%cbr_f_{i}#0")
        feed("f32", f"%cbr_f_{i}#1")
    for i in range(2):
        lines.append(f"  %cbr_idx_{i}:2 = fabric.instance @pe_cond_br_index({port('i1')}, {port('index')}) "
                     f'{{sym_name = "cbr_idx_{i}"}} : (i1, index) -> (index, index)')
        feed("index", f"%cbr_idx_{i}#0")
        feed("index", f"%cbr_idx_{i}#1")

    # --- Mux PEs ---
    lines.append("  // Mux PEs")
    for i in range(4):
        lines.append(f"  %mux_i_{i} = fabric.instance @pe_mux_i32({port('index')}, {port('i32')}, {port('i32')}) "
                     f'{{sym_name = "mux_i_{i}"}} : (index, i32, i32) -> i32')
        feed("i32", f"%mux_i_{i}")
    for i in range(6):
        lines.append(f"  %mux_n_{i} = fabric.instance @pe_mux_none({port('index')}, {port('none')}, {port('none')}) "
                     f'{{sym_name = "mux_n_{i}"}} : (index, none, none) -> none')
        feed("none", f"%mux_n_{i}")
    for i in range(2):
        lines.append(f"  %mux_idx_{i} = fabric.instance @pe_mux_index({port('index')}, {port('index')}, {port('index')}) "
                     f'{{sym_name = "mux_idx_{i}"}} : (index, index, index) -> index')
        feed("index", f"%mux_idx_{i}")
    for i in range(2):
        lines.append(f"  %mux_f_{i} = fabric.instance @pe_mux_f32({port('index')}, {port('f32')}, {port('f32')}) "
                     f'{{sym_name = "mux_f_{i}"}} : (index, f32, f32) -> f32')
        feed("f32", f"%mux_f_{i}")
    for i in range(2):
        lines.append(f"  %mux_i64_{i} = fabric.instance @pe_mux_i64({port('index')}, {port('i64')}, {port('i64')}) "
                     f'{{sym_name = "mux_i64_{i}"}} : (index, i64, i64) -> i64')
        feed("i64", f"%mux_i64_{i}")

    # --- Memory PEs ---
    lines.append("  // Load/Store PEs")
    for i in range(4):
        lines.append(f"  %load_{i}:2 = fabric.instance @pe_load({port('index')}, {port('i32')}, {port('none')}) "
                     f'{{sym_name = "load_{i}"}} : (index, i32, none) -> (i32, index)')
        feed("i32", f"%load_{i}#0")
        feed("index", f"%load_{i}#1")
    for i in range(2):
        lines.append(f"  %store_{i}:2 = fabric.instance @pe_store({port('index')}, {port('i32')}, {port('none')}) "
                     f'{{sym_name = "store_{i}"}} : (index, i32, none) -> (i32, index)')
        feed("i32", f"%store_{i}#0")
        feed("index", f"%store_{i}#1")
    for i in range(2):
        lines.append(f"  %load_f32_{i}:2 = fabric.instance @pe_load_f32({port('index')}, {port('f32')}, {port('none')}) "
                     f'{{sym_name = "load_f32_{i}"}} : (index, f32, none) -> (f32, index)')
        feed("f32", f"%load_f32_{i}#0")
        feed("index", f"%load_f32_{i}#1")
    for i in range(2):
        lines.append(f"  %store_f32_{i}:2 = fabric.instance @pe_store_f32({port('index')}, {port('f32')}, {port('none')}) "
                     f'{{sym_name = "store_f32_{i}"}} : (index, f32, none) -> (f32, index)')
        feed("f32", f"%store_f32_{i}#0")
        feed("index", f"%store_f32_{i}#1")

    # --- Sink PEs ---
    lines.append("  // Sink PEs")
    for i in range(5):
        lines.append(f"  fabric.instance @pe_sink_i1({port('i1')}) "
                     f'{{sym_name = "sink_i1_{i}"}} : (i1) -> ()')
    for i in range(2):
        lines.append(f"  fabric.instance @pe_sink_none({port('none')}) "
                     f'{{sym_name = "sink_n_{i}"}} : (none) -> ()')
    for i in range(2):
        lines.append(f"  fabric.instance @pe_sink_i32({port('i32')}) "
                     f'{{sym_name = "sink_i32_{i}"}} : (i32) -> ()')
    for i in range(2):
        lines.append(f"  fabric.instance @pe_sink_index({port('index')}) "
                     f'{{sym_name = "sink_idx_{i}"}} : (index) -> ()')
    for i in range(2):
        lines.append(f"  fabric.instance @pe_sink_f32({port('f32')}) "
                     f'{{sym_name = "sink_f32_{i}"}} : (f32) -> ()')
    for i in range(2):
        lines.append(f"  fabric.instance @pe_sink_i64({port('i64')}) "
                     f'{{sym_name = "sink_i64_{i}"}} : (i64) -> ()')

    # --- External memory ---
    lines.append("  // External memory interfaces")
    # extmem_i32_0: 1 load, 0 store
    lines.append(f"  %extmem_i32_0_o0, %extmem_i32_0_o1 = fabric.extmemory")
    lines.append(f"    [ldCount = 1, stCount = 0]")
    lines.append(f"    (%mem_i32_0, {port('index')})")
    lines.append(f"    : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, index) -> (i32, none)")
    feed("i32", "%extmem_i32_0_o0")
    feed("none", "%extmem_i32_0_o1")

    # extmem_i32_1: 1 load, 0 store
    lines.append(f"  %extmem_i32_1_o0, %extmem_i32_1_o1 = fabric.extmemory")
    lines.append(f"    [ldCount = 1, stCount = 0]")
    lines.append(f"    (%mem_i32_1, {port('index')})")
    lines.append(f"    : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, index) -> (i32, none)")
    feed("i32", "%extmem_i32_1_o0")
    feed("none", "%extmem_i32_1_o1")

    # extmem_i32_2: 0 load, 1 store (write-only)
    lines.append(f"  %extmem_i32_2_o0 = fabric.extmemory")
    lines.append(f"    [ldCount = 0, stCount = 1, lsqDepth = 4]")
    lines.append(f"    (%mem_i32_2, {port('index')}, {port('i32')})")
    lines.append(f"    : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, index, i32) -> (none)")
    feed("none", "%extmem_i32_2_o0")

    # extmem_i32_3: 1 load, 1 store (read-write)
    # Port order: memref, ldaddr(index), staddr(index), stdata(i32)
    # Outputs: lddata(i32), lddone(none), stdone(none)
    lines.append(f"  %extmem_i32_3_o0, %extmem_i32_3_o1, %extmem_i32_3_o2 = fabric.extmemory")
    lines.append(f"    [ldCount = 1, stCount = 1, lsqDepth = 4]")
    lines.append(f"    (%mem_i32_3, {port('index')}, {port('index')}, {port('i32')})")
    lines.append(f"    : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, index, index, i32) -> (i32, none, none)")
    feed("i32", "%extmem_i32_3_o0")
    feed("none", "%extmem_i32_3_o1")
    feed("none", "%extmem_i32_3_o2")

    # extmem_i32_4: 1 load, 0 store (extra read-only i32)
    lines.append(f"  %extmem_i32_4_o0, %extmem_i32_4_o1 = fabric.extmemory")
    lines.append(f"    [ldCount = 1, stCount = 0]")
    lines.append(f"    (%mem_i32_4, {port('index')})")
    lines.append(f"    : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, index) -> (i32, none)")
    feed("i32", "%extmem_i32_4_o0")
    feed("none", "%extmem_i32_4_o1")

    # extmem_f32_0: 1 load, 0 store
    lines.append(f"  %extmem_f32_0_o0, %extmem_f32_0_o1 = fabric.extmemory")
    lines.append(f"    [ldCount = 1, stCount = 0]")
    lines.append(f"    (%mem_f32_0, {port('index')})")
    lines.append(f"    : memref<?xf32, strided<[1], offset: ?>>, (memref<?xf32, strided<[1], offset: ?>>, index) -> (f32, none)")
    feed("f32", "%extmem_f32_0_o0")
    feed("none", "%extmem_f32_0_o1")

    # extmem_f32_1: 0 load, 1 store (write-only f32)
    lines.append(f"  %extmem_f32_1_o0 = fabric.extmemory")
    lines.append(f"    [ldCount = 0, stCount = 1, lsqDepth = 4]")
    lines.append(f"    (%mem_f32_1, {port('index')}, {port('f32')})")
    lines.append(f"    : memref<?xf32, strided<[1], offset: ?>>, (memref<?xf32, strided<[1], offset: ?>>, index, f32) -> (none)")
    feed("none", "%extmem_f32_1_o0")

    # extmem_f32_2: 1 load, 0 store (extra read-only f32)
    lines.append(f"  %extmem_f32_2_o0, %extmem_f32_2_o1 = fabric.extmemory")
    lines.append(f"    [ldCount = 1, stCount = 0]")
    lines.append(f"    (%mem_f32_2, {port('index')})")
    lines.append(f"    : memref<?xf32, strided<[1], offset: ?>>, (memref<?xf32, strided<[1], offset: ?>>, index) -> (f32, none)")
    feed("f32", "%extmem_f32_2_o0")
    feed("none", "%extmem_f32_2_o1")

    # extmem_f32_3: 1 load, 1 store (read-write f32)
    # Port order: memref, ldaddr(index), staddr(index), stdata(f32)
    # Outputs: lddata(f32), lddone(none), stdone(none)
    lines.append(f"  %extmem_f32_3_o0, %extmem_f32_3_o1, %extmem_f32_3_o2 = fabric.extmemory")
    lines.append(f"    [ldCount = 1, stCount = 1, lsqDepth = 4]")
    lines.append(f"    (%mem_f32_3, {port('index')}, {port('index')}, {port('f32')})")
    lines.append(f"    : memref<?xf32, strided<[1], offset: ?>>, (memref<?xf32, strided<[1], offset: ?>>, index, index, f32) -> (f32, none, none)")
    feed("f32", "%extmem_f32_3_o0")
    feed("none", "%extmem_f32_3_o1")
    feed("none", "%extmem_f32_3_o2")

    # extmem_f32_4: 0 load, 1 store (extra write-only f32)
    lines.append(f"  %extmem_f32_4_o0 = fabric.extmemory")
    lines.append(f"    [ldCount = 0, stCount = 1, lsqDepth = 4]")
    lines.append(f"    (%mem_f32_4, {port('index')}, {port('f32')})")
    lines.append(f"    : memref<?xf32, strided<[1], offset: ?>>, (memref<?xf32, strided<[1], offset: ?>>, index, f32) -> (none)")
    feed("none", "%extmem_f32_4_o0")

    # On-chip scratchpad memory (fabric.memory)
    lines.append("  // On-chip scratchpad memory")
    # mem_f32_scratch_0: 1 load, 1 store (f32 scratchpad)
    lines.append(f"  %mem_f32_s0_o0, %mem_f32_s0_o1, %mem_f32_s0_o2 = fabric.memory")
    lines.append(f"    [ldCount = 1, stCount = 1, lsqDepth = 4, numRegion = 1,")
    lines.append(f"     addr_offset_table = array<i64: 1, 0, 1, 0>]")
    lines.append(f"    ({port('index')}, {port('index')}, {port('f32')})")
    lines.append(f"    : memref<256xf32>, (index, index, f32) -> (f32, none, none)")
    feed("f32", "%mem_f32_s0_o0")
    feed("none", "%mem_f32_s0_o1")
    feed("none", "%mem_f32_s0_o2")

    # mem_f32_scratch_1: 1 load, 1 store (second f32 scratchpad)
    lines.append(f"  %mem_f32_s1_o0, %mem_f32_s1_o1, %mem_f32_s1_o2 = fabric.memory")
    lines.append(f"    [ldCount = 1, stCount = 1, lsqDepth = 4, numRegion = 1,")
    lines.append(f"     addr_offset_table = array<i64: 1, 0, 1, 0>]")
    lines.append(f"    ({port('index')}, {port('index')}, {port('f32')})")
    lines.append(f"    : memref<256xf32>, (index, index, f32) -> (f32, none, none)")
    feed("f32", "%mem_f32_s1_o0")
    feed("none", "%mem_f32_s1_o1")
    feed("none", "%mem_f32_s1_o2")

    # mem_i32_scratch: 1 load, 1 store (i32 scratchpad)
    lines.append(f"  %mem_i32_s_o0, %mem_i32_s_o1, %mem_i32_s_o2 = fabric.memory")
    lines.append(f"    [ldCount = 1, stCount = 1, lsqDepth = 4, numRegion = 1,")
    lines.append(f"     addr_offset_table = array<i64: 1, 0, 1, 0>]")
    lines.append(f"    ({port('index')}, {port('index')}, {port('i32')})")
    lines.append(f"    : memref<256xi32>, (index, index, i32) -> (i32, none, none)")
    feed("i32", "%mem_i32_s_o0")
    feed("none", "%mem_i32_s_o1")
    feed("none", "%mem_i32_s_o2")

    # mem_i32_ro: 1 load, 0 store (read-only i32 scratchpad)
    lines.append(f"  %mem_i32_ro_o0, %mem_i32_ro_o1 = fabric.memory")
    lines.append(f"    [ldCount = 1, stCount = 0, numRegion = 1,")
    lines.append(f"     addr_offset_table = array<i64: 1, 0, 1, 0>]")
    lines.append(f"    ({port('index')})")
    lines.append(f"    : memref<256xi32>, (index) -> (i32, none)")
    feed("i32", "%mem_i32_ro_o0")
    feed("none", "%mem_i32_ro_o1")

    # Module output ports: i32, f32, none to cover all return types
    o_i32 = port("i32")
    o_f32 = port("f32")
    o_none = port("none")
    lines.append("")
    lines.append(f"  fabric.yield {o_i32}, {o_f32}, {o_none} : i32, f32, none")

    # Now generate the switch declarations and module header
    # Module inputs feed into switches
    mod_inputs_i32 = ["%in0", "%in1", "%in2", "%in3"]
    mod_inputs_i64 = ["%in_i64_0"]
    mod_inputs_none = ["%ctrl_in"]
    mod_inputs_index = ["%addr0"]

    for v in mod_inputs_i32:
        feed("i32", v)
    for v in mod_inputs_i64:
        feed("i64", v)
    for v in mod_inputs_none:
        feed("none", v)
    for v in mod_inputs_index:
        feed("index", v)

    # Now build the actual output
    out.write("fabric.module @loom_cgra_small(\n")
    out.write("    %mem_i32_0: memref<?xi32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_i32_1: memref<?xi32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_i32_2: memref<?xi32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_i32_3: memref<?xi32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_i32_4: memref<?xi32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_f32_0: memref<?xf32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_f32_1: memref<?xf32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_f32_2: memref<?xf32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_f32_3: memref<?xf32, strided<[1], offset: ?>>,\n")
    out.write("    %mem_f32_4: memref<?xf32, strided<[1], offset: ?>>,\n")
    out.write("    %ctrl_in: none, %in0: i32, %in1: i32, %in2: i32, %in3: i32,\n")
    out.write("    %in_i64_0: i64, %addr0: index\n")
    out.write(") -> (i32, f32, none) {\n\n")

    # Generate switch declarations
    for typ in ["i32", "f32", "i64", "index", "i1", "none"]:
        p = planes[typ]
        tname = {"i32": "i32", "f32": "f32", "i64": "i64",
                 "index": "idx", "i1": "i1", "none": "n"}[typ]
        n_in = p.n_in
        n_out = p.n_out
        ct = full_ct(n_in, n_out)

        # Build input list
        in_names = ", ".join(p.inputs)
        # Build type annotations
        out_types = ", ".join([typ] * n_out)

        out.write(f"  %csw_{tname}:{n_out} = fabric.switch "
                  f"[connectivity_table = [{ct}]]\n")
        out.write(f"    {in_names}\n")
        out.write(f"    : {typ} -> {', '.join([typ] * n_out)}\n\n")

    # Write instance lines
    for line in lines:
        out.write(line + "\n")

    out.write("}\n")


def main():
    out = sys.stdout

    out.write("// Small CGRA template for apps with <= 400 handshake lines\n")
    out.write("//\n")
    out.write("// Architecture: central-switch routing per type plane.\n")
    out.write("\n")
    out.write("module {\n\n")

    out.write(PE_DEFS)
    out.write("\n")

    generate_module(out)

    out.write("\n}\n")


if __name__ == "__main__":
    main()
