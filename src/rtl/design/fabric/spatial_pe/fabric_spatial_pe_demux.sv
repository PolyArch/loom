// fabric_spatial_pe_demux.sv -- Output demux bank for spatial PE.
//
// Provides per-FU-output demultiplexers that route FU outputs to the
// PE's external output ports.  Each demux has an independent
// sel/discard/disconnect configuration field loaded from the PE's
// config register.
//
// For each of NUM_FU_OUT FU outputs:
//   - sel        selects which PE output to drive (SEL_WIDTH bits)
//   - discard    drains the FU output (ready=1) without driving any
//                PE output valid
//   - disconnect severs the route: FU output ready=0, no PE output
//                driven by this entry

module fabric_spatial_pe_demux
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_FU_OUT  = 2,
  parameter int unsigned NUM_PE_OUT  = 4,
  parameter int unsigned DATA_WIDTH  = 32,
  parameter int unsigned SEL_WIDTH   = clog2(NUM_PE_OUT)
)(
  // --- FU-level output handshake (from the FU) ---
  input  logic [NUM_FU_OUT-1:0]    fu_out_valid,
  output logic [NUM_FU_OUT-1:0]    fu_out_ready,
  input  logic [DATA_WIDTH-1:0]    fu_out_data  [NUM_FU_OUT],

  // --- PE-level output valid/data contribution from this demux bank ---
  // Each entry drives at most one PE output.  The top level must
  // OR-combine demux_pe_out_valid with other sources.
  output logic [NUM_PE_OUT-1:0]    demux_pe_out_valid,
  output logic [DATA_WIDTH-1:0]    demux_pe_out_data  [NUM_PE_OUT],

  // --- PE-level output ready (directly from PE ports) ---
  input  logic [NUM_PE_OUT-1:0]    pe_out_ready,

  // --- Per-demux config fields (directly from config register) ---
  input  logic [SEL_WIDTH > 0 ? SEL_WIDTH-1 : 0 : 0]
                                   sel_cfg        [NUM_FU_OUT],
  input  logic [NUM_FU_OUT-1:0]   discard_cfg,
  input  logic [NUM_FU_OUT-1:0]   disconnect_cfg
);

  // Effective sel register width: at least 1 bit to avoid zero-width.
  localparam int unsigned SEL_W = (SEL_WIDTH > 0) ? SEL_WIDTH : 1;

  // ---------------------------------------------------------------
  // PE output valid and data drive
  //
  // Each PE output is driven by at most one FU output (the one whose
  // sel matches).  Since only one FU is active at a time, at most
  // one FU output will target any given PE output.
  // ---------------------------------------------------------------
  always_comb begin : pe_output_drive
    integer iter_var0;
    integer iter_var1;

    // Default: no PE output driven.
    for (iter_var0 = 0; iter_var0 < NUM_PE_OUT; iter_var0 = iter_var0 + 1) begin : pe_out_init
      demux_pe_out_valid[iter_var0] = 1'b0;
      demux_pe_out_data[iter_var0]  = '0;
    end : pe_out_init

    // Route each FU output to its selected PE output.
    for (iter_var0 = 0; iter_var0 < NUM_FU_OUT; iter_var0 = iter_var0 + 1) begin : per_fu_out
      if (!disconnect_cfg[iter_var0] && !discard_cfg[iter_var0]) begin : mode_normal
        for (iter_var1 = 0; iter_var1 < NUM_PE_OUT; iter_var1 = iter_var1 + 1) begin : scan_pe_out
          if (SEL_W'(iter_var1) == sel_cfg[iter_var0]) begin : match
            demux_pe_out_valid[iter_var1] = fu_out_valid[iter_var0];
            demux_pe_out_data[iter_var1]  = fu_out_data[iter_var0];
          end : match
        end : scan_pe_out
      end : mode_normal
    end : per_fu_out
  end : pe_output_drive

  // ---------------------------------------------------------------
  // FU output ready generation
  //
  // - disconnect: FU output ready = 0 (stall the FU)
  // - discard:    FU output ready = 1 (drain locally)
  // - normal:     FU output ready = ready of the selected PE output
  // ---------------------------------------------------------------
  always_comb begin : fu_ready_drive
    integer iter_var0;
    integer iter_var1;

    for (iter_var0 = 0; iter_var0 < NUM_FU_OUT; iter_var0 = iter_var0 + 1) begin : per_fu_out_rdy
      if (disconnect_cfg[iter_var0]) begin : rdy_disconnect
        fu_out_ready[iter_var0] = 1'b0;
      end : rdy_disconnect
      else if (discard_cfg[iter_var0]) begin : rdy_discard
        fu_out_ready[iter_var0] = 1'b1;
      end : rdy_discard
      else begin : rdy_normal
        fu_out_ready[iter_var0] = 1'b0;
        for (iter_var1 = 0; iter_var1 < NUM_PE_OUT; iter_var1 = iter_var1 + 1) begin : scan_pe_out_rdy
          if (SEL_W'(iter_var1) == sel_cfg[iter_var0]) begin : match
            fu_out_ready[iter_var0] = pe_out_ready[iter_var1];
          end : match
        end : scan_pe_out_rdy
      end : rdy_normal
    end : per_fu_out_rdy
  end : fu_ready_drive

endmodule : fabric_spatial_pe_demux
