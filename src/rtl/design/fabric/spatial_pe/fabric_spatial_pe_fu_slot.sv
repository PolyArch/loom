// fabric_spatial_pe_fu_slot.sv -- Per-FU gating wrapper for spatial PE.
//
// Gates FU input valid signals and output ready signals based on
// whether this FU's index matches the currently configured opcode.
// When inactive (opcode mismatch or PE disabled), all FU inputs are
// driven invalid and all FU output readies are deasserted, ensuring
// no spurious computation occurs.
//
// Also extracts this FU's config bits from the shared FU config region
// of the PE's configuration register.

module fabric_spatial_pe_fu_slot
  import fabric_pkg::*;
#(
  parameter int unsigned FU_INDEX    = 0,
  parameter int unsigned DATA_WIDTH  = 32,
  parameter int unsigned NUM_FU_IN   = 2,
  parameter int unsigned NUM_FU_OUT  = 1,
  parameter int unsigned FU_CFG_BITS = 0,
  // Opcode width from PE level (clog2(NUM_FU)).
  parameter int unsigned OPCODE_WIDTH = 1
)(
  // --- PE enable and opcode from config register ---
  input  logic                     pe_enable,
  input  logic [OPCODE_WIDTH > 0 ? OPCODE_WIDTH-1 : 0 : 0]
                                   pe_opcode,

  // --- Ungated FU input handshake (from mux bank) ---
  input  logic [NUM_FU_IN-1:0]    ungated_in_valid,
  output logic [NUM_FU_IN-1:0]    ungated_in_ready,
  input  logic [DATA_WIDTH-1:0]   ungated_in_data   [NUM_FU_IN],

  // --- Gated FU input handshake (towards FU body) ---
  output logic [NUM_FU_IN-1:0]    fu_in_valid,
  input  logic [NUM_FU_IN-1:0]    fu_in_ready,
  output logic [DATA_WIDTH-1:0]   fu_in_data        [NUM_FU_IN],

  // --- FU output handshake (from FU body) ---
  input  logic [NUM_FU_OUT-1:0]   fu_out_valid,
  output logic [NUM_FU_OUT-1:0]   fu_out_ready,
  input  logic [DATA_WIDTH-1:0]   fu_out_data       [NUM_FU_OUT],

  // --- Gated FU output handshake (towards demux bank) ---
  output logic [NUM_FU_OUT-1:0]   gated_out_valid,
  input  logic [NUM_FU_OUT-1:0]   gated_out_ready,
  output logic [DATA_WIDTH-1:0]   gated_out_data    [NUM_FU_OUT],

  // --- Shared FU config region from PE config register ---
  // Width is MAX_FU_CFG_BITS from the PE level.  This slot extracts
  // the first FU_CFG_BITS bits relevant to this FU.
  input  logic [FU_CFG_BITS > 0 ? FU_CFG_BITS-1 : 0 : 0]
                                   fu_cfg_bits,

  // --- FU-specific config bits output (towards FU body) ---
  output logic [FU_CFG_BITS > 0 ? FU_CFG_BITS-1 : 0 : 0]
                                   fu_cfg_out
);

  // Effective opcode width: at least 1 bit.
  localparam int unsigned OPC_W = (OPCODE_WIDTH > 0) ? OPCODE_WIDTH : 1;

  // ---------------------------------------------------------------
  // Active detection
  // ---------------------------------------------------------------
  logic fu_active;
  assign fu_active = pe_enable &&
                     (OPC_W'(FU_INDEX) == pe_opcode);

  // ---------------------------------------------------------------
  // Input gating: when inactive, FU inputs are not valid and
  // ungated inputs are not accepted.
  // ---------------------------------------------------------------
  always_comb begin : input_gate
    integer iter_var0;

    for (iter_var0 = 0; iter_var0 < NUM_FU_IN; iter_var0 = iter_var0 + 1) begin : per_in
      if (fu_active) begin : active_in
        fu_in_valid[iter_var0]     = ungated_in_valid[iter_var0];
        fu_in_data[iter_var0]      = ungated_in_data[iter_var0];
        ungated_in_ready[iter_var0] = fu_in_ready[iter_var0];
      end : active_in
      else begin : inactive_in
        fu_in_valid[iter_var0]     = 1'b0;
        fu_in_data[iter_var0]      = '0;
        ungated_in_ready[iter_var0] = 1'b0;
      end : inactive_in
    end : per_in
  end : input_gate

  // ---------------------------------------------------------------
  // Output gating: when inactive, FU outputs are not valid and
  // the FU body sees no downstream ready.
  // ---------------------------------------------------------------
  always_comb begin : output_gate
    integer iter_var0;

    for (iter_var0 = 0; iter_var0 < NUM_FU_OUT; iter_var0 = iter_var0 + 1) begin : per_out
      if (fu_active) begin : active_out
        gated_out_valid[iter_var0] = fu_out_valid[iter_var0];
        gated_out_data[iter_var0]  = fu_out_data[iter_var0];
        fu_out_ready[iter_var0]    = gated_out_ready[iter_var0];
      end : active_out
      else begin : inactive_out
        gated_out_valid[iter_var0] = 1'b0;
        gated_out_data[iter_var0]  = '0;
        fu_out_ready[iter_var0]    = 1'b0;
      end : inactive_out
    end : per_out
  end : output_gate

  // ---------------------------------------------------------------
  // FU config routing: pass through the relevant bits.
  // When FU_CFG_BITS == 0, the 1-bit dummy port is tied to zero.
  // ---------------------------------------------------------------
  assign fu_cfg_out = fu_cfg_bits;

endmodule : fabric_spatial_pe_fu_slot
