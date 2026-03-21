// Standalone filelist for Wave 7 testbenches.
// Excludes FP behavioral modules that use shortreal/$bitstoreal
// (not supported by Verilator in all modes).
// Includes only modules needed by: tb_backpressure_test (fifo),
// tb_config_equiv_test (add_tag), tb_memory_sideeffect_test (memory).

// Packages (must be first)
common/fabric_pkg.sv
common/fabric_axi_pkg.sv

// Interfaces
common/fabric_handshake_if.sv
common/fabric_cfg_if.sv

// Common primitives
common/fabric_rr_arbiter.sv
common/fabric_fifo_mem.sv
common/fabric_broadcast_tracker.sv

// Fabric leaf modules used by standalone TBs
fabric/fabric_add_tag.sv
fabric/fabric_del_tag.sv
fabric/fabric_fifo.sv
fabric/fabric_config_ctrl.sv

// Memory modules
fabric/memory/fabric_memory.sv
fabric/memory/fabric_memory_lsq.sv
fabric/memory/fabric_memory_sram.sv
