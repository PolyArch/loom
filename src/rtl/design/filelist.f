// Filelist for standalone fabric RTL design compilation.
// Compilation order: packages, interfaces, common leaf modules,
// then design modules grouped by category.

// --- Packages ---
common/fabric_pkg.sv
common/fabric_axi_pkg.sv

// --- Interfaces ---
common/fabric_handshake_if.sv
common/fabric_cfg_if.sv

// --- Common leaf modules ---
common/fabric_fifo_mem.sv
common/fabric_rr_arbiter.sv
common/fabric_broadcast_tracker.sv

// --- Arith FU ops ---
arith/fu_op_addi.sv
arith/fu_op_subi.sv
arith/fu_op_andi.sv
arith/fu_op_ori.sv
arith/fu_op_xori.sv
arith/fu_op_shli.sv
arith/fu_op_shrsi.sv
arith/fu_op_shrui.sv
arith/fu_op_cmpi.sv
arith/fu_op_cmpf.sv
arith/fu_op_extsi.sv
arith/fu_op_extui.sv
arith/fu_op_trunci.sv
arith/fu_op_select.sv
arith/fu_op_index_cast.sv
arith/fu_op_index_castui.sv
arith/fu_op_muli.sv
arith/fu_op_divsi.sv
arith/fu_op_divui.sv
arith/fu_op_remsi.sv
arith/fu_op_remui.sv
arith/fu_op_addf.sv
arith/fu_op_subf.sv
arith/fu_op_mulf.sv
arith/fu_op_divf.sv
arith/fu_op_negf.sv
arith/fu_op_fptosi.sv
arith/fu_op_fptoui.sv
arith/fu_op_sitofp.sv
arith/fu_op_uitofp.sv

// --- Dataflow FU ops ---
dataflow/fu_op_carry.sv
dataflow/fu_op_gate.sv
dataflow/fu_op_invariant.sv
dataflow/fu_op_stream.sv

// --- Handshake FU ops ---
handshake/fu_op_cond_br.sv
handshake/fu_op_constant.sv
handshake/fu_op_join.sv
handshake/fu_op_load.sv
handshake/fu_op_mux.sv
handshake/fu_op_store.sv

// --- LLVM FU ops ---
llvm/fu_op_bitreverse.sv

// --- Math FU ops ---
math/fu_op_absf.sv
math/fu_op_cos.sv
math/fu_op_exp.sv
math/fu_op_fma.sv
math/fu_op_log2.sv
math/fu_op_sin.sv
math/fu_op_sqrt.sv

// --- Fabric infrastructure modules ---
fabric/fabric_add_tag.sv
fabric/fabric_config_ctrl.sv
fabric/fabric_del_tag.sv
fabric/fabric_fifo.sv
fabric/fabric_map_tag.sv
fabric/fabric_mux.sv

// --- Fabric external memory ---
fabric/extmemory/fabric_extmemory_req.sv
fabric/extmemory/fabric_extmemory_resp.sv
fabric/extmemory/fabric_extmemory.sv

// --- Fabric memory ---
fabric/memory/fabric_memory_lsq.sv
fabric/memory/fabric_memory_sram.sv
fabric/memory/fabric_memory.sv

// --- Fabric spatial PE ---
fabric/spatial_pe/fabric_spatial_pe_demux.sv
fabric/spatial_pe/fabric_spatial_pe_fu_slot.sv
fabric/spatial_pe/fabric_spatial_pe_mux.sv
fabric/spatial_pe/fabric_spatial_pe.sv

// --- Fabric spatial switch ---
fabric/spatial_sw/fabric_spatial_sw_core.sv
fabric/spatial_sw/fabric_spatial_sw_decomp.sv
fabric/spatial_sw/fabric_spatial_sw.sv

// --- Fabric temporal PE ---
fabric/temporal_pe/fabric_temporal_pe_fu_slot.sv
fabric/temporal_pe/fabric_temporal_pe_imem.sv
fabric/temporal_pe/fabric_temporal_pe_operand.sv
fabric/temporal_pe/fabric_temporal_pe_output_arb.sv
fabric/temporal_pe/fabric_temporal_pe_regfile.sv
fabric/temporal_pe/fabric_temporal_pe_scheduler.sv
fabric/temporal_pe/fabric_temporal_pe.sv

// --- Fabric temporal switch ---
fabric/temporal_sw/fabric_temporal_sw_arbiter.sv
fabric/temporal_sw/fabric_temporal_sw_slot_match.sv
fabric/temporal_sw/fabric_temporal_sw.sv
