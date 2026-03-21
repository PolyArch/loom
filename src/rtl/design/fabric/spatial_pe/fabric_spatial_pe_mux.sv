// fabric_spatial_pe_mux.sv -- Input mux bank for spatial PE.
//
// Provides per-FU-input multiplexers that select from the PE's external
// input ports.  Each mux has an independent sel/discard/disconnect
// configuration field loaded from the PE's config register.
//
// For each of NUM_FU_IN FU inputs:
//   - sel       selects which PE input to route (SEL_WIDTH bits)
//   - discard   consumes the selected PE input (ready=1) but drives
//               FU input valid=0
//   - disconnect severs the route: FU input valid=0, PE input ready
//               not affected by this mux entry

module fabric_spatial_pe_mux
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_PE_IN  = 4,
  parameter int unsigned NUM_FU_IN  = 2,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned SEL_WIDTH  = clog2(NUM_PE_IN)
)(
  // --- PE-level input data (directly from PE ports) ---
  input  logic [NUM_PE_IN-1:0]     pe_in_valid,
  input  logic [DATA_WIDTH-1:0]    pe_in_data  [NUM_PE_IN],

  // --- PE-level input ready contribution from this mux bank ---
  // Each bit is OR-contributed into the PE-level ready.
  // The top level must OR mux_pe_in_ready with other sources.
  output logic [NUM_PE_IN-1:0]     mux_pe_in_ready,

  // --- FU-level input handshake (towards the FU) ---
  output logic [NUM_FU_IN-1:0]     fu_in_valid,
  input  logic [NUM_FU_IN-1:0]     fu_in_ready,
  output logic [DATA_WIDTH-1:0]    fu_in_data  [NUM_FU_IN],

  // --- Per-mux config fields (directly from config register) ---
  // sel_cfg:        selection index per FU input
  // discard_cfg:    discard flag per FU input
  // disconnect_cfg: disconnect flag per FU input
  input  logic [SEL_WIDTH > 0 ? SEL_WIDTH-1 : 0 : 0]
                                   sel_cfg        [NUM_FU_IN],
  input  logic [NUM_FU_IN-1:0]    discard_cfg,
  input  logic [NUM_FU_IN-1:0]    disconnect_cfg
);

  // Effective sel register width: at least 1 bit to avoid zero-width.
  localparam int unsigned SEL_W = (SEL_WIDTH > 0) ? SEL_WIDTH : 1;

  // ---------------------------------------------------------------
  // FU input data and valid selection
  // ---------------------------------------------------------------
  always_comb begin : fu_input_drive
    integer iter_var0;
    integer iter_var1;

    for (iter_var0 = 0; iter_var0 < NUM_FU_IN; iter_var0 = iter_var0 + 1) begin : per_fu_in
      fu_in_valid[iter_var0] = 1'b0;
      fu_in_data[iter_var0]  = '0;

      if (disconnect_cfg[iter_var0]) begin : mode_disconnect
        // FU input is inert.
        fu_in_valid[iter_var0] = 1'b0;
        fu_in_data[iter_var0]  = '0;
      end : mode_disconnect
      else if (discard_cfg[iter_var0]) begin : mode_discard
        // Selected PE input consumed, but FU input not driven valid.
        fu_in_valid[iter_var0] = 1'b0;
        fu_in_data[iter_var0]  = '0;
      end : mode_discard
      else begin : mode_normal
        // Route selected PE input to FU input.
        for (iter_var1 = 0; iter_var1 < NUM_PE_IN; iter_var1 = iter_var1 + 1) begin : scan_pe_in
          if (SEL_W'(iter_var1) == sel_cfg[iter_var0]) begin : match
            fu_in_valid[iter_var0] = pe_in_valid[iter_var1];
            fu_in_data[iter_var0]  = pe_in_data[iter_var1];
          end : match
        end : scan_pe_in
      end : mode_normal
    end : per_fu_in
  end : fu_input_drive

  // ---------------------------------------------------------------
  // PE input ready contribution
  //
  // A PE input is marked ready by this mux bank if at least one
  // FU input selects it AND the FU input is ready (or discard mode).
  // Disconnect entries never contribute ready.
  // ---------------------------------------------------------------
  always_comb begin : pe_ready_drive
    integer iter_var0;
    integer iter_var1;

    for (iter_var0 = 0; iter_var0 < NUM_PE_IN; iter_var0 = iter_var0 + 1) begin : per_pe_in
      mux_pe_in_ready[iter_var0] = 1'b0;

      for (iter_var1 = 0; iter_var1 < NUM_FU_IN; iter_var1 = iter_var1 + 1) begin : scan_fu_in
        if (!disconnect_cfg[iter_var1] &&
            SEL_W'(iter_var0) == sel_cfg[iter_var1]) begin : sel_match
          if (discard_cfg[iter_var1]) begin : rdy_discard
            // Discard: always accept from PE input.
            mux_pe_in_ready[iter_var0] = 1'b1;
          end : rdy_discard
          else begin : rdy_normal
            // Normal: PE input ready when FU input is ready.
            mux_pe_in_ready[iter_var0] = fu_in_ready[iter_var1];
          end : rdy_normal
        end : sel_match
      end : scan_fu_in
    end : per_pe_in
  end : pe_ready_drive

endmodule : fabric_spatial_pe_mux
