#!/usr/bin/env python3
"""Generate CGRA fabric templates for mapper-app tests.

Produces all 3 template tiers with comprehensive PE catalog covering
all operations used by quick-tier apps.

Usage:
    python3 gen_templates.py small  > loom_cgra_small.fabric.mlir
    python3 gen_templates.py medium > loom_cgra_medium.fabric.mlir
    python3 gen_templates.py large  > loom_cgra_large.fabric.mlir
"""

import sys

# ---------------------------------------------------------------------------
# PE definition catalog
# Each entry: (pe_name, input_types, output_types, mlir_body, latency_hw)
# latency_hw is the 3rd latency element (1st/2nd always 1)
# ---------------------------------------------------------------------------

PE_DEFS = [
    # --- Constants ---
    ("pe_const_i32", ["none"], ["i32"],
     "%r = handshake.constant %a0 {value = 0 : i32} : i32", 1),
    ("pe_const_f32", ["none"], ["f32"],
     "%r = handshake.constant %a0 {value = 0.000000e+00 : f32} : f32", 1),
    ("pe_const_i64", ["none"], ["i64"],
     "%r = handshake.constant %a0 {value = 0 : i64} : i64", 1),
    ("pe_const_index", ["none"], ["index"],
     "%r = handshake.constant %a0 {value = 0 : index} : index", 1),
    ("pe_const_i1", ["none"], ["i1"],
     "%r = handshake.constant %a0 {value = true} : i1", 1),

    # --- Joins ---
    ("pe_join1", ["none"], ["none"],
     "%r = handshake.join %a0 : none", 1),
    ("pe_join2", ["none", "none"], ["none"],
     "%r = handshake.join %a0, %a1 : none, none", 1),
    ("pe_join3", ["none", "none", "none"], ["none"],
     "%r = handshake.join %a0, %a1, %a2 : none, none, none", 1),
    ("pe_join4", ["none", "none", "none", "none"], ["none"],
     "%r = handshake.join %a0, %a1, %a2, %a3 : none, none, none, none", 1),
    ("pe_join_i1", ["i1"], ["none"],
     "%r = handshake.join %a0 : i1", 1),

    # --- Cond branch ---
    ("pe_cond_br_none", ["i1", "none"], ["none", "none"],
     "%t, %f = handshake.cond_br %a0, %a1 : none", 1),
    ("pe_cond_br_i32", ["i1", "i32"], ["i32", "i32"],
     "%t, %f = handshake.cond_br %a0, %a1 : i32", 1),
    ("pe_cond_br_f32", ["i1", "f32"], ["f32", "f32"],
     "%t, %f = handshake.cond_br %a0, %a1 : f32", 1),
    ("pe_cond_br_index", ["i1", "index"], ["index", "index"],
     "%t, %f = handshake.cond_br %a0, %a1 : index", 1),

    # --- Mux ---
    ("pe_mux_i32", ["index", "i32", "i32"], ["i32"],
     "%r = handshake.mux %a0 [%a1, %a2] : index, i32", 1),
    ("pe_mux_f32", ["index", "f32", "f32"], ["f32"],
     "%r = handshake.mux %a0 [%a1, %a2] : index, f32", 1),
    ("pe_mux_none", ["index", "none", "none"], ["none"],
     "%r = handshake.mux %a0 [%a1, %a2] : index, none", 1),
    ("pe_mux_index", ["index", "index", "index"], ["index"],
     "%r = handshake.mux %a0 [%a1, %a2] : index, index", 1),

    # --- Integer arithmetic (i32) ---
    ("pe_addi", ["i32", "i32"], ["i32"],
     "%r = arith.addi %a0, %a1 : i32", 1),
    ("pe_subi", ["i32", "i32"], ["i32"],
     "%r = arith.subi %a0, %a1 : i32", 1),
    ("pe_muli", ["i32", "i32"], ["i32"],
     "%r = arith.muli %a0, %a1 : i32", 2),
    ("pe_divui", ["i32", "i32"], ["i32"],
     "%r = arith.divui %a0, %a1 : i32", 10),
    ("pe_divsi", ["i32", "i32"], ["i32"],
     "%r = arith.divsi %a0, %a1 : i32", 10),
    ("pe_remui", ["i32", "i32"], ["i32"],
     "%r = arith.remui %a0, %a1 : i32", 10),
    ("pe_remsi", ["i32", "i32"], ["i32"],
     "%r = arith.remsi %a0, %a1 : i32", 10),

    # --- Integer arithmetic (i64) ---
    ("pe_addi_i64", ["i64", "i64"], ["i64"],
     "%r = arith.addi %a0, %a1 : i64", 1),
    ("pe_subi_i64", ["i64", "i64"], ["i64"],
     "%r = arith.subi %a0, %a1 : i64", 1),
    ("pe_muli_i64", ["i64", "i64"], ["i64"],
     "%r = arith.muli %a0, %a1 : i64", 2),
    ("pe_cmpi_i64", ["i64", "i64"], ["i1"],
     "%r = arith.cmpi ult, %a0, %a1 : i64", 1),
    ("pe_shli_i64", ["i64", "i64"], ["i64"],
     "%r = arith.shli %a0, %a1 : i64", 1),
    ("pe_remui_i64", ["i64", "i64"], ["i64"],
     "%r = arith.remui %a0, %a1 : i64", 10),

    # --- Integer arithmetic (index) ---
    ("pe_addi_index", ["index", "index"], ["index"],
     "%r = arith.addi %a0, %a1 : index", 1),
    ("pe_divui_index", ["index", "index"], ["index"],
     "%r = arith.divui %a0, %a1 : index", 10),
    ("pe_remui_index", ["index", "index"], ["index"],
     "%r = arith.remui %a0, %a1 : index", 10),
    ("pe_muli_index", ["index", "index"], ["index"],
     "%r = arith.muli %a0, %a1 : index", 2),

    # --- Bitwise (i32) ---
    ("pe_andi", ["i32", "i32"], ["i32"],
     "%r = arith.andi %a0, %a1 : i32", 1),
    ("pe_ori", ["i32", "i32"], ["i32"],
     "%r = arith.ori %a0, %a1 : i32", 1),
    ("pe_xori", ["i32", "i32"], ["i32"],
     "%r = arith.xori %a0, %a1 : i32", 1),
    ("pe_shli", ["i32", "i32"], ["i32"],
     "%r = arith.shli %a0, %a1 : i32", 1),
    ("pe_shrui", ["i32", "i32"], ["i32"],
     "%r = arith.shrui %a0, %a1 : i32", 1),
    ("pe_shrsi", ["i32", "i32"], ["i32"],
     "%r = arith.shrsi %a0, %a1 : i32", 1),

    # --- Bitwise (i1) ---
    ("pe_xori_i1", ["i1", "i1"], ["i1"],
     "%r = arith.xori %a0, %a1 : i1", 1),

    # --- Float arithmetic ---
    ("pe_addf", ["f32", "f32"], ["f32"],
     "%r = arith.addf %a0, %a1 : f32", 3),
    ("pe_subf", ["f32", "f32"], ["f32"],
     "%r = arith.subf %a0, %a1 : f32", 3),
    ("pe_mulf", ["f32", "f32"], ["f32"],
     "%r = arith.mulf %a0, %a1 : f32", 3),
    ("pe_divf", ["f32", "f32"], ["f32"],
     "%r = arith.divf %a0, %a1 : f32", 10),
    ("pe_fma", ["f32", "f32", "f32"], ["f32"],
     "%r = math.fma %a0, %a1, %a2 : f32", 3),

    # --- Float unary ---
    ("pe_negf", ["f32"], ["f32"],
     "%r = arith.negf %a0 : f32", 1),
    ("pe_absf", ["f32"], ["f32"],
     "%r = math.absf %a0 : f32", 1),
    ("pe_sin", ["f32"], ["f32"],
     "%r = math.sin %a0 : f32", 10),
    ("pe_cos", ["f32"], ["f32"],
     "%r = math.cos %a0 : f32", 10),
    ("pe_exp", ["f32"], ["f32"],
     "%r = math.exp %a0 : f32", 10),
    ("pe_sqrt", ["f32"], ["f32"],
     "%r = math.sqrt %a0 : f32", 10),
    ("pe_log2", ["f32"], ["f32"],
     "%r = math.log2 %a0 : f32", 10),

    # --- Compare ---
    ("pe_cmpi", ["i32", "i32"], ["i1"],
     "%r = arith.cmpi ult, %a0, %a1 : i32", 1),
    ("pe_cmpf", ["f32", "f32"], ["i1"],
     "%r = arith.cmpf ult, %a0, %a1 : f32", 1),

    # --- Select ---
    ("pe_select", ["i1", "i32", "i32"], ["i32"],
     "%r = arith.select %a0, %a1, %a2 : i32", 1),
    ("pe_select_index", ["i1", "index", "index"], ["index"],
     "%r = arith.select %a0, %a1, %a2 : index", 1),
    ("pe_select_f32", ["i1", "f32", "f32"], ["f32"],
     "%r = arith.select %a0, %a1, %a2 : f32", 1),

    # --- Type cast ---
    ("pe_index_cast_i32", ["i32"], ["index"],
     "%r = arith.index_cast %a0 : i32 to index", 1),
    ("pe_index_cast_i64", ["i64"], ["index"],
     "%r = arith.index_cast %a0 : i64 to index", 1),
    ("pe_index_cast_to_i32", ["index"], ["i32"],
     "%r = arith.index_cast %a0 : index to i32", 1),
    ("pe_index_cast_to_i64", ["index"], ["i64"],
     "%r = arith.index_cast %a0 : index to i64", 1),
    ("pe_index_castui", ["i32"], ["index"],
     "%r = arith.index_castui %a0 : i32 to index", 1),
    # i32 <-> i64
    ("pe_extui", ["i32"], ["i64"],
     "%r = arith.extui %a0 : i32 to i64", 1),
    ("pe_trunci", ["i64"], ["i32"],
     "%r = arith.trunci %a0 : i64 to i32", 1),
    # i1 <-> i32
    ("pe_extui_i1", ["i1"], ["i32"],
     "%r = arith.extui %a0 : i1 to i32", 1),
    ("pe_trunci_to_i1", ["i32"], ["i1"],
     "%r = arith.trunci %a0 : i32 to i1", 1),
    # i16 conversions
    ("pe_extui_i16", ["i16"], ["i32"],
     "%r = arith.extui %a0 : i16 to i32", 1),
    ("pe_trunci_to_i16", ["i64"], ["i16"],
     "%r = arith.trunci %a0 : i64 to i16", 1),
    ("pe_remui_i16", ["i16", "i16"], ["i16"],
     "%r = arith.remui %a0, %a1 : i16", 10),
    ("pe_const_i16", ["none"], ["i16"],
     "%r = handshake.constant %a0 {value = 0 : i16} : i16", 1),
    ("pe_uitofp_i16", ["i16"], ["f32"],
     "%r = arith.uitofp %a0 : i16 to f32", 3),
    # int <-> float
    ("pe_uitofp", ["i32"], ["f32"],
     "%r = arith.uitofp %a0 : i32 to f32", 3),
    ("pe_sitofp", ["i32"], ["f32"],
     "%r = arith.sitofp %a0 : i32 to f32", 3),
    ("pe_fptoui", ["f32"], ["i32"],
     "%r = arith.fptoui %a0 : f32 to i32", 3),

    # --- Dataflow ---
    ("pe_stream", ["index", "index", "index"], ["index", "i1"],
     "%idx, %wc = dataflow.stream %a0, %a1, %a2", 1),
    ("pe_gate", ["i32", "i1"], ["i32", "i1"],
     "%av, %ac = dataflow.gate %a0, %a1 : i32, i1 -> i32, i1", 1),
    ("pe_gate_f32", ["f32", "i1"], ["f32", "i1"],
     "%av, %ac = dataflow.gate %a0, %a1 : f32, i1 -> f32, i1", 1),
    ("pe_gate_index", ["index", "i1"], ["index", "i1"],
     "%av, %ac = dataflow.gate %a0, %a1 : index, i1 -> index, i1", 1),
    ("pe_carry", ["i1", "i32", "i32"], ["i32"],
     "%o = dataflow.carry %a0, %a1, %a2 : i1, i32, i32 -> i32", 1),
    ("pe_carry_f32", ["i1", "f32", "f32"], ["f32"],
     "%o = dataflow.carry %a0, %a1, %a2 : i1, f32, f32 -> f32", 1),
    ("pe_carry_none", ["i1", "none", "none"], ["none"],
     "%o = dataflow.carry %a0, %a1, %a2 : i1, none, none -> none", 1),
    ("pe_carry_index", ["i1", "index", "index"], ["index"],
     "%o = dataflow.carry %a0, %a1, %a2 : i1, index, index -> index", 1),
    ("pe_invariant", ["i1", "i32"], ["i32"],
     "%o = dataflow.invariant %a0, %a1 : i1, i32 -> i32", 1),
    ("pe_invariant_i1", ["i1", "i1"], ["i1"],
     "%o = dataflow.invariant %a0, %a1 : i1, i1 -> i1", 1),
    ("pe_invariant_none", ["i1", "none"], ["none"],
     "%o = dataflow.invariant %a0, %a1 : i1, none -> none", 1),
    ("pe_invariant_f32", ["i1", "f32"], ["f32"],
     "%o = dataflow.invariant %a0, %a1 : i1, f32 -> f32", 1),
    ("pe_invariant_index", ["i1", "index"], ["index"],
     "%o = dataflow.invariant %a0, %a1 : i1, index -> index", 1),

    # --- Memory ---
    ("pe_load", ["index", "i32", "none"], ["i32", "index"],
     "%ld_d, %ld_a = handshake.load [%a0] %a1, %a2 : index, i32", 1),
    ("pe_load_f32", ["index", "f32", "none"], ["f32", "index"],
     "%ld_d, %ld_a = handshake.load [%a0] %a1, %a2 : index, f32", 1),
    ("pe_store", ["index", "i32", "none"], ["i32", "index"],
     "%st_d, %st_a = handshake.store [%a0] %a1, %a2 : index, i32", 1),
    ("pe_store_f32", ["index", "f32", "none"], ["f32", "index"],
     "%st_d, %st_a = handshake.store [%a0] %a1, %a2 : index, f32", 1),

    # --- Sink ---
    ("pe_sink_i1", ["i1"], [],
     "handshake.sink %a0 : i1", 1),
    ("pe_sink_none", ["none"], [],
     "handshake.sink %a0 : none", 1),
    ("pe_sink_i32", ["i32"], [],
     "handshake.sink %a0 : i32", 1),
    ("pe_sink_index", ["index"], [],
     "handshake.sink %a0 : index", 1),
    ("pe_sink_f32", ["f32"], [],
     "handshake.sink %a0 : f32", 1),
]

PE_DEF_MAP = {d[0]: d for d in PE_DEFS}

# ---------------------------------------------------------------------------
# Instance counts per template tier
# ---------------------------------------------------------------------------

SMALL_INSTANCES = {
    "pe_const_i32": 4, "pe_const_f32": 2, "pe_const_i64": 3,
    "pe_const_index": 5, "pe_const_i1": 2, "pe_const_i16": 1,
    "pe_join1": 4, "pe_join2": 1, "pe_join3": 1, "pe_join4": 1,
    "pe_join_i1": 1,
    "pe_cond_br_none": 5, "pe_cond_br_i32": 2, "pe_cond_br_f32": 1,
    "pe_cond_br_index": 1,
    "pe_mux_i32": 2, "pe_mux_f32": 1, "pe_mux_none": 3,
    "pe_mux_index": 1,
    # i32 arith
    "pe_addi": 4, "pe_subi": 2, "pe_muli": 2, "pe_divui": 1,
    "pe_divsi": 1, "pe_remui": 1, "pe_remsi": 1,
    # i64 arith
    "pe_addi_i64": 2, "pe_subi_i64": 1, "pe_muli_i64": 1,
    "pe_cmpi_i64": 2, "pe_shli_i64": 1, "pe_remui_i64": 1,
    # index arith
    "pe_addi_index": 1, "pe_divui_index": 1,
    "pe_remui_index": 1, "pe_muli_index": 1,
    # bitwise
    "pe_andi": 1, "pe_ori": 1, "pe_xori": 1,
    "pe_shli": 1, "pe_shrui": 1, "pe_shrsi": 1,
    "pe_xori_i1": 1,
    # float
    "pe_addf": 1, "pe_subf": 1, "pe_mulf": 1, "pe_divf": 1,
    "pe_fma": 1,
    "pe_negf": 1, "pe_absf": 1, "pe_sin": 1, "pe_cos": 1,
    "pe_exp": 1, "pe_sqrt": 1, "pe_log2": 1,
    # compare
    "pe_cmpi": 3, "pe_cmpf": 1,
    # select
    "pe_select": 2, "pe_select_index": 3, "pe_select_f32": 1,
    # type cast
    "pe_index_cast_i32": 5, "pe_index_cast_i64": 3,
    "pe_index_cast_to_i32": 1, "pe_index_cast_to_i64": 2,
    "pe_index_castui": 1,
    "pe_extui": 4, "pe_trunci": 2,
    "pe_extui_i1": 3, "pe_trunci_to_i1": 2,
    "pe_extui_i16": 1, "pe_trunci_to_i16": 1, "pe_remui_i16": 1,
    "pe_uitofp": 1, "pe_uitofp_i16": 1, "pe_sitofp": 1, "pe_fptoui": 1,
    # dataflow
    "pe_stream": 2, "pe_gate": 3, "pe_gate_f32": 1,
    "pe_gate_index": 3,
    "pe_carry": 4, "pe_carry_f32": 1, "pe_carry_none": 4,
    "pe_carry_index": 1,
    "pe_invariant": 4, "pe_invariant_i1": 2, "pe_invariant_none": 2,
    "pe_invariant_f32": 1, "pe_invariant_index": 1,
    # memory
    "pe_load": 3, "pe_load_f32": 1, "pe_store": 2, "pe_store_f32": 1,
    # sink
    "pe_sink_i1": 3, "pe_sink_none": 2, "pe_sink_i32": 1,
    "pe_sink_index": 1, "pe_sink_f32": 1,
}

MEDIUM_INSTANCES = {
    "pe_const_i32": 6, "pe_const_f32": 3, "pe_const_i64": 4,
    "pe_const_index": 12, "pe_const_i1": 3, "pe_const_i16": 1,
    "pe_join1": 6, "pe_join2": 2, "pe_join3": 2, "pe_join4": 1,
    "pe_join_i1": 2,
    "pe_cond_br_none": 14, "pe_cond_br_i32": 3, "pe_cond_br_f32": 2,
    "pe_cond_br_index": 1,
    "pe_mux_i32": 4, "pe_mux_f32": 2, "pe_mux_none": 8,
    "pe_mux_index": 2,
    "pe_addi": 8, "pe_subi": 3, "pe_muli": 5, "pe_divui": 2,
    "pe_divsi": 1, "pe_remui": 2, "pe_remsi": 1,
    "pe_addi_i64": 3, "pe_subi_i64": 1, "pe_muli_i64": 1,
    "pe_cmpi_i64": 3, "pe_shli_i64": 1, "pe_remui_i64": 1,
    "pe_addi_index": 1, "pe_divui_index": 1,
    "pe_remui_index": 1, "pe_muli_index": 1,
    "pe_andi": 3, "pe_ori": 2, "pe_xori": 2,
    "pe_shli": 2, "pe_shrui": 2, "pe_shrsi": 1,
    "pe_xori_i1": 1,
    "pe_addf": 3, "pe_subf": 3, "pe_mulf": 3, "pe_divf": 2,
    "pe_fma": 3,
    "pe_negf": 2, "pe_absf": 1, "pe_sin": 1, "pe_cos": 1,
    "pe_exp": 1, "pe_sqrt": 1, "pe_log2": 1,
    "pe_cmpi": 6, "pe_cmpf": 2,
    "pe_select": 4, "pe_select_index": 6, "pe_select_f32": 1,
    "pe_index_cast_i32": 12, "pe_index_cast_i64": 5,
    "pe_index_cast_to_i32": 2, "pe_index_cast_to_i64": 3,
    "pe_index_castui": 2,
    "pe_extui": 8, "pe_trunci": 4,
    "pe_extui_i1": 4, "pe_trunci_to_i1": 3,
    "pe_extui_i16": 1, "pe_trunci_to_i16": 1, "pe_remui_i16": 1,
    "pe_uitofp": 2, "pe_uitofp_i16": 1, "pe_sitofp": 1, "pe_fptoui": 1,
    "pe_stream": 4, "pe_gate": 6, "pe_gate_f32": 2,
    "pe_gate_index": 5,
    "pe_carry": 8, "pe_carry_f32": 2, "pe_carry_none": 10,
    "pe_carry_index": 1,
    "pe_invariant": 10, "pe_invariant_i1": 3, "pe_invariant_none": 3,
    "pe_invariant_f32": 2, "pe_invariant_index": 1,
    "pe_load": 5, "pe_load_f32": 3, "pe_store": 3, "pe_store_f32": 2,
    "pe_sink_i1": 4, "pe_sink_none": 3, "pe_sink_i32": 2,
    "pe_sink_index": 1, "pe_sink_f32": 1,
}

LARGE_INSTANCES = {
    "pe_const_i32": 8, "pe_const_f32": 4, "pe_const_i64": 5,
    "pe_const_index": 16, "pe_const_i1": 4, "pe_const_i16": 1,
    "pe_join1": 6, "pe_join2": 2, "pe_join3": 2, "pe_join4": 2,
    "pe_join_i1": 2,
    "pe_cond_br_none": 18, "pe_cond_br_i32": 5, "pe_cond_br_f32": 3,
    "pe_cond_br_index": 2,
    "pe_mux_i32": 8, "pe_mux_f32": 3, "pe_mux_none": 10,
    "pe_mux_index": 3,
    "pe_addi": 12, "pe_subi": 3, "pe_muli": 8, "pe_divui": 3,
    "pe_divsi": 2, "pe_remui": 2, "pe_remsi": 1,
    "pe_addi_i64": 3, "pe_subi_i64": 1, "pe_muli_i64": 2,
    "pe_cmpi_i64": 4, "pe_shli_i64": 2, "pe_remui_i64": 1,
    "pe_addi_index": 2, "pe_divui_index": 2,
    "pe_remui_index": 1, "pe_muli_index": 1,
    "pe_andi": 3, "pe_ori": 2, "pe_xori": 2,
    "pe_shli": 2, "pe_shrui": 2, "pe_shrsi": 1,
    "pe_xori_i1": 1,
    "pe_addf": 4, "pe_subf": 3, "pe_mulf": 4, "pe_divf": 2,
    "pe_fma": 3,
    "pe_negf": 2, "pe_absf": 2, "pe_sin": 1, "pe_cos": 1,
    "pe_exp": 1, "pe_sqrt": 1, "pe_log2": 1,
    "pe_cmpi": 8, "pe_cmpf": 2,
    "pe_select": 8, "pe_select_index": 8, "pe_select_f32": 2,
    "pe_index_cast_i32": 12, "pe_index_cast_i64": 5,
    "pe_index_cast_to_i32": 3, "pe_index_cast_to_i64": 3,
    "pe_index_castui": 2,
    "pe_extui": 8, "pe_trunci": 5,
    "pe_extui_i1": 5, "pe_trunci_to_i1": 3,
    "pe_extui_i16": 1, "pe_trunci_to_i16": 1, "pe_remui_i16": 1,
    "pe_uitofp": 2, "pe_uitofp_i16": 1, "pe_sitofp": 1, "pe_fptoui": 1,
    "pe_stream": 5, "pe_gate": 8, "pe_gate_f32": 3,
    "pe_gate_index": 6,
    "pe_carry": 10, "pe_carry_f32": 3, "pe_carry_none": 10,
    "pe_carry_index": 2,
    "pe_invariant": 10, "pe_invariant_i1": 3, "pe_invariant_none": 3,
    "pe_invariant_f32": 2, "pe_invariant_index": 2,
    "pe_load": 5, "pe_load_f32": 3, "pe_store": 3, "pe_store_f32": 2,
    "pe_sink_i1": 5, "pe_sink_none": 3, "pe_sink_i32": 2,
    "pe_sink_index": 2, "pe_sink_f32": 2,
}

MODULE_CONFIGS = {
    "small": {
        "instances": SMALL_INSTANCES,
        "module_name": "loom_cgra_small",
        "n_mem_i32": 1,
        "n_mem_f32": 0,
        "n_in_i32": 4,
        "n_in_index": 1,
        "n_out_i32": 1,
        "n_out_none": 1,
        "n_privmem_i32": 1,
        "n_privmem_f32": 1,
    },
    "medium": {
        "instances": MEDIUM_INSTANCES,
        "module_name": "loom_cgra_medium",
        "n_mem_i32": 2,
        "n_mem_f32": 2,
        "n_in_i32": 4,
        "n_in_index": 1,
        "n_out_i32": 1,
        "n_out_f32": 1,
        "n_out_none": 1,
        "n_privmem_i32": 1,
        "n_privmem_f32": 2,
    },
    "large": {
        "instances": LARGE_INSTANCES,
        "module_name": "loom_cgra_large",
        "n_mem_i32": 3,
        "n_mem_f32": 2,
        "n_in_i32": 4,
        "n_in_index": 1,
        "n_out_i32": 1,
        "n_out_f32": 1,
        "n_out_none": 1,
        "n_privmem_i32": 2,
        "n_privmem_f32": 2,
    },
}

TYPE_PLANES = ["i32", "f32", "index", "i1", "none", "i64", "i16"]


def emit_pe_defs(out, pe_names):
    """Emit fabric.pe definitions for all PE types used."""
    for pe_name in pe_names:
        d = PE_DEF_MAP[pe_name]
        _, in_types, out_types, body, lat_hw = d
        args = ", ".join(f"%a{i}: {t}" for i, t in enumerate(in_types))
        lat = f"[1 : i16, 1 : i16, {lat_hw} : i16]"
        intvl = "[1 : i16, 1 : i16, 1 : i16]"
        out_sig = ", ".join(out_types)
        out.write(f"fabric.pe @{pe_name}({args})")
        out.write(f" [latency = {lat}, interval = {intvl}]")
        out.write(f" -> ({out_sig}) {{\n")
        out.write(f"  {body}\n")
        if out_types:
            results = extract_results(body, out_types)
            out.write(f"  fabric.yield {results}\n")
        else:
            out.write("  fabric.yield\n")
        out.write("}\n")


def extract_results(body, out_types):
    """Extract result variable names from body for fabric.yield."""
    if not out_types:
        return ""
    first_line = body.strip().split("\n")[0].strip()
    if "=" in first_line:
        lhs = first_line.split("=")[0].strip()
        types_str = ", ".join(out_types)
        return f"{lhs} : {types_str}"
    return ""


class SwitchBuilder:
    """Tracks inputs and outputs for a type-plane switch."""

    def __init__(self, type_name):
        self.type_name = type_name
        self.inputs = []
        self.out_idx = 0

    def add_input(self, value_name):
        self.inputs.append(value_name)

    def alloc_output(self):
        idx = self.out_idx
        self.out_idx += 1
        return idx

    @property
    def n_in(self):
        return len(self.inputs)

    @property
    def n_out(self):
        return self.out_idx


def generate_template(tier, out):
    """Generate a complete CGRA template for the given tier."""
    cfg = MODULE_CONFIGS[tier]
    instances = cfg["instances"]
    pe_types_used = sorted(set(instances.keys()))

    # Header
    label = {"small": "<= 300", "medium": "301-800", "large": "> 800"}[tier]
    out.write(f"// {tier.title()} CGRA template for apps with {label} "
              f"handshake lines\n")
    out.write("//\n// Architecture: central-switch routing per type plane.\n")
    out.write("\nmodule {\n\n")

    emit_pe_defs(out, pe_types_used)
    out.write("\n")

    # Build switch I/O tracking
    switches = {t: SwitchBuilder(t) for t in TYPE_PLANES}
    pe_inst_info = []

    # Module inputs
    n_in_i32 = cfg["n_in_i32"]
    n_in_index = cfg.get("n_in_index", 0)
    for i in range(n_in_i32):
        switches["i32"].add_input(f"%in{i}")
    for i in range(n_in_index):
        switches["index"].add_input(f"%addr{i}")
    switches["none"].add_input("%ctrl_in")

    # Extmemory outputs -> switch inputs
    n_mem_i32 = cfg["n_mem_i32"]
    n_mem_f32 = cfg.get("n_mem_f32", 0)
    for i in range(n_mem_i32):
        switches["i32"].add_input(f"%extmem_i32_{i}_ld")
        switches["none"].add_input(f"%extmem_i32_{i}_done")
    for i in range(n_mem_f32):
        switches["f32"].add_input(f"%extmem_f32_{i}_ld")
        switches["none"].add_input(f"%extmem_f32_{i}_done")

    # Private memory outputs -> switch inputs
    n_priv_i32 = cfg.get("n_privmem_i32", 0)
    n_priv_f32 = cfg.get("n_privmem_f32", 0)
    for i in range(n_priv_i32):
        switches["i32"].add_input(f"%privmem_i32_{i}_ld")
        switches["none"].add_input(f"%privmem_i32_{i}_ld_done")
        switches["none"].add_input(f"%privmem_i32_{i}_st_done")
    for i in range(n_priv_f32):
        switches["f32"].add_input(f"%privmem_f32_{i}_ld")
        switches["none"].add_input(f"%privmem_f32_{i}_ld_done")
        switches["none"].add_input(f"%privmem_f32_{i}_st_done")

    # PE instances
    for pe_name in pe_types_used:
        count = instances.get(pe_name, 0)
        d = PE_DEF_MAP[pe_name]
        _, in_types, out_types, _, _ = d

        for idx in range(count):
            sym = f"{pe_name.replace('pe_', '')}_{idx}"
            input_refs = []
            for in_type in in_types:
                sw = switches[in_type]
                out_slot = sw.alloc_output()
                input_refs.append((in_type, out_slot))

            output_names = []
            if len(out_types) == 0:
                pass
            elif len(out_types) == 1:
                vname = f"%{sym}"
                switches[out_types[0]].add_input(vname)
                output_names.append(vname)
            else:
                for oi, ot in enumerate(out_types):
                    vname = f"%{sym}#{oi}"
                    switches[ot].add_input(vname)
                    output_names.append(vname)

            pe_inst_info.append((sym, pe_name, input_refs, output_names,
                                 in_types, out_types))

    # Module output slots
    n_out_i32 = cfg.get("n_out_i32", 0)
    n_out_f32 = cfg.get("n_out_f32", 0)
    n_out_none = cfg.get("n_out_none", 0)
    mod_out_slots = []
    for _ in range(n_out_i32):
        mod_out_slots.append(("i32", switches["i32"].alloc_output()))
    for _ in range(n_out_f32):
        mod_out_slots.append(("f32", switches["f32"].alloc_output()))
    for _ in range(n_out_none):
        mod_out_slots.append(("none", switches["none"].alloc_output()))

    # Extmemory input slots
    extmem_i32_slots = []
    for i in range(n_mem_i32):
        extmem_i32_slots.append({"addr": switches["index"].alloc_output()})
    extmem_f32_slots = []
    for i in range(n_mem_f32):
        extmem_f32_slots.append({"addr": switches["index"].alloc_output()})

    # Private memory input slots
    privmem_i32_slots = []
    for i in range(n_priv_i32):
        privmem_i32_slots.append({
            "ld_addr": switches["index"].alloc_output(),
            "st_addr": switches["index"].alloc_output(),
            "st_data": switches["i32"].alloc_output(),
        })
    privmem_f32_slots = []
    for i in range(n_priv_f32):
        privmem_f32_slots.append({
            "ld_addr": switches["index"].alloc_output(),
            "st_addr": switches["index"].alloc_output(),
            "st_data": switches["f32"].alloc_output(),
        })

    # --- Emit module ---
    mod_args = []
    for i in range(n_mem_i32):
        mod_args.append(
            f"%mem_i32_{i}: memref<?xi32, strided<[1], offset: ?>>")
    for i in range(n_mem_f32):
        mod_args.append(
            f"%mem_f32_{i}: memref<?xf32, strided<[1], offset: ?>>")
    mod_args.append("%ctrl_in: none")
    for i in range(n_in_i32):
        mod_args.append(f"%in{i}: i32")
    for i in range(n_in_index):
        mod_args.append(f"%addr{i}: index")

    mod_out_types = [t for t, _ in mod_out_slots]
    out.write(f"fabric.module @{cfg['module_name']}(\n")
    out.write("    " + ", ".join(mod_args) + "\n")
    out.write(f") -> ({', '.join(mod_out_types)}) {{\n\n")

    # Emit switches
    for tp in TYPE_PLANES:
        sw = switches[tp]
        if sw.n_in == 0 or sw.n_out == 0:
            continue
        ct_size = sw.n_in * sw.n_out
        ct = ", ".join(["1"] * ct_size)
        in_list = ",\n      ".join(sw.inputs)
        out_types_str = ", ".join([tp] * sw.n_out)

        out.write(f"  // Central switch: {tp} "
                  f"({sw.n_in} -> {sw.n_out})\n")
        out.write(f"  %csw_{tp}:{sw.n_out} = fabric.switch "
                  f"[connectivity_table = [{ct}]]\n")
        out.write(f"      {in_list}\n")
        out.write(f"      : {tp} -> {out_types_str}\n\n")

    # Emit PE instances
    out.write("  // PE instances\n")
    for sym, pe_name, input_refs, _, in_types, out_types in pe_inst_info:
        in_refs = [f"%csw_{t}#{s}" for t, s in input_refs]
        in_refs_str = ", ".join(in_refs)
        in_types_str = ", ".join(in_types)

        if len(out_types) == 0:
            out.write(f"  fabric.instance @{pe_name}({in_refs_str})\n")
            out.write(f"    {{sym_name = \"{sym}\"}} "
                      f": ({in_types_str}) -> ()\n")
        elif len(out_types) == 1:
            out.write(f"  %{sym} = fabric.instance "
                      f"@{pe_name}({in_refs_str})\n")
            out.write(f"    {{sym_name = \"{sym}\"}} "
                      f": ({in_types_str}) -> {out_types[0]}\n")
        else:
            ot_str = ", ".join(out_types)
            out.write(f"  %{sym}:{len(out_types)} = fabric.instance "
                      f"@{pe_name}({in_refs_str})\n")
            out.write(f"    {{sym_name = \"{sym}\"}} "
                      f": ({in_types_str}) -> ({ot_str})\n")

    # Emit extmemory
    out.write("\n  // External memory\n")
    for i in range(n_mem_i32):
        addr_ref = f"%csw_index#{extmem_i32_slots[i]['addr']}"
        out.write(f"  %extmem_i32_{i}_ld, %extmem_i32_{i}_done = "
                  f"fabric.extmemory\n")
        out.write(f"    [ldCount = 1, stCount = 1, lsqDepth = 4]\n")
        out.write(f"    (%mem_i32_{i}, {addr_ref})\n")
        out.write(f"    : memref<?xi32, strided<[1], offset: ?>>, "
                  f"(memref<?xi32, strided<[1], offset: ?>>, index) "
                  f"-> (i32, none)\n")
    for i in range(n_mem_f32):
        addr_ref = f"%csw_index#{extmem_f32_slots[i]['addr']}"
        out.write(f"  %extmem_f32_{i}_ld, %extmem_f32_{i}_done = "
                  f"fabric.extmemory\n")
        out.write(f"    [ldCount = 1, stCount = 1, lsqDepth = 4]\n")
        out.write(f"    (%mem_f32_{i}, {addr_ref})\n")
        out.write(f"    : memref<?xf32, strided<[1], offset: ?>>, "
                  f"(memref<?xf32, strided<[1], offset: ?>>, index) "
                  f"-> (f32, none)\n")

    # Emit private memory
    if n_priv_i32 > 0 or n_priv_f32 > 0:
        out.write("\n  // Private memory (handshake.memory)\n")
    for i in range(n_priv_i32):
        s = privmem_i32_slots[i]
        out.write(f"  %privmem_i32_{i}_ld, %privmem_i32_{i}_ld_done, "
                  f"%privmem_i32_{i}_st_done = fabric.memory\n")
        out.write(f"    [ldCount = 1, stCount = 1, lsqDepth = 4, "
                  f"is_private = true]\n")
        out.write(f"    (%csw_index#{s['ld_addr']}, "
                  f"%csw_index#{s['st_addr']}, "
                  f"%csw_i32#{s['st_data']})\n")
        out.write(f"    : memref<1024xi32>, (index, index, i32) "
                  f"-> (i32, none, none)\n")
    for i in range(n_priv_f32):
        s = privmem_f32_slots[i]
        out.write(f"  %privmem_f32_{i}_ld, %privmem_f32_{i}_ld_done, "
                  f"%privmem_f32_{i}_st_done = fabric.memory\n")
        out.write(f"    [ldCount = 1, stCount = 1, lsqDepth = 4, "
                  f"is_private = true]\n")
        out.write(f"    (%csw_index#{s['ld_addr']}, "
                  f"%csw_index#{s['st_addr']}, "
                  f"%csw_f32#{s['st_data']})\n")
        out.write(f"    : memref<1024xf32>, (index, index, f32) "
                  f"-> (f32, none, none)\n")

    # Module yield
    yield_refs = [f"%csw_{t}#{s}" for t, s in mod_out_slots]
    yield_types = ", ".join(mod_out_types)
    out.write(f"\n  fabric.yield {', '.join(yield_refs)} : {yield_types}\n")
    out.write("}\n\n}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: gen_templates.py <small|medium|large>", file=sys.stderr)
        sys.exit(1)
    tier = sys.argv[1]
    if tier not in MODULE_CONFIGS:
        print(f"Unknown tier: {tier}", file=sys.stderr)
        sys.exit(1)
    generate_template(tier, sys.stdout)


if __name__ == "__main__":
    main()
