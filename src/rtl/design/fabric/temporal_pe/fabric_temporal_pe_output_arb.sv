// fabric_temporal_pe_output_arb.sv -- Output arbitration (Layer 3).
//
// Each FU has per-output FU-local output registers.  No bypass is
// allowed: every FU completion must first write to these registers.
//
// When multiple FUs have valid output registers targeting the same PE
// output port, round-robin arbitration by FU definition order selects
// the winner.  One fabric_rr_arbiter per PE output port handles this.
//
// This module also handles:
//   - Result tag writeback: winning FU's result_tag is driven on the
//     PE output tag field
//   - Register writeback: if a result config has is_reg set, the FU
//     output is routed to the register file write port instead of the
//     PE output
//   - Output register drain: clears the FU-local output register when
//     the output is successfully arbitrated and transferred

module fabric_temporal_pe_output_arb
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_FU       = 2,
  parameter int unsigned MAX_FU_OUT   = 2,
  parameter int unsigned NUM_PE_OUT   = 4,
  parameter int unsigned DATA_WIDTH   = 32,
  parameter int unsigned TAG_WIDTH    = 4,
  parameter int unsigned NUM_REG      = 0
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Per-FU pipeline outputs (from fu_slot wrappers) ---
  // pipe_valid[fu][out]: pipeline output valid for FU fu, output out
  input  logic                     pipe_valid    [0:NUM_FU-1][0:MAX_FU_OUT-1],
  input  logic [DATA_WIDTH-1:0]    pipe_data     [0:NUM_FU-1][0:MAX_FU_OUT-1],

  // --- Per-FU output register status (back to fu_slot busy logic) ---
  output logic                     out_reg_occupied [0:NUM_FU-1][0:MAX_FU_OUT-1],

  // --- Per-FU active slot info (from top PE, valid only when FU has pending results) ---
  // fu_active[fu]: this FU has an active instruction context
  input  logic [NUM_FU-1:0]       fu_active,
  // Output demux config for each FU's active slot
  input  logic [(NUM_PE_OUT > 1 ? $clog2(NUM_PE_OUT) : 1)-1:0]
                                   fu_out_demux_sel        [0:NUM_FU-1][0:MAX_FU_OUT-1],
  input  logic                     fu_out_demux_discard    [0:NUM_FU-1][0:MAX_FU_OUT-1],
  input  logic                     fu_out_demux_disconnect [0:NUM_FU-1][0:MAX_FU_OUT-1],
  // Result config for each FU's active slot
  input  logic [TAG_WIDTH-1:0]    fu_result_tag           [0:NUM_FU-1][0:MAX_FU_OUT-1],
  input  logic                     fu_result_is_reg        [0:NUM_FU-1][0:MAX_FU_OUT-1],
  input  logic [(NUM_REG > 1 ? $clog2(NUM_REG) : 1)-1:0]
                                   fu_result_reg_idx       [0:NUM_FU-1][0:MAX_FU_OUT-1],

  // --- PE output ports ---
  output logic [NUM_PE_OUT-1:0]   pe_out_valid,
  output logic [DATA_WIDTH-1:0]   pe_out_data   [NUM_PE_OUT],
  output logic [TAG_WIDTH-1:0]    pe_out_tag    [NUM_PE_OUT],
  input  logic [NUM_PE_OUT-1:0]   pe_out_ready,

  // --- Register writeback port ---
  output logic [MAX_FU_OUT-1:0]   reg_wr_enable,
  output logic [(NUM_REG > 1 ? $clog2(NUM_REG) : 1)-1:0]
                                   reg_wr_idx     [MAX_FU_OUT],
  output logic [DATA_WIDTH-1:0]   reg_wr_data    [MAX_FU_OUT]
);

  // ---------------------------------------------------------------
  // Derived widths
  // ---------------------------------------------------------------
  localparam int unsigned OUT_SEL_W  = clog2(NUM_PE_OUT);
  localparam int unsigned REG_IDX_W  = clog2(NUM_REG);
  localparam int unsigned FU_IDX_W   = (NUM_FU > 1) ? $clog2(NUM_FU) : 1;
  localparam int unsigned EFF_OUT_SEL = (OUT_SEL_W > 0) ? OUT_SEL_W : 1;
  localparam int unsigned EFF_REG_IDX = (REG_IDX_W > 0) ? REG_IDX_W : 1;

  // ---------------------------------------------------------------
  // FU-local output registers
  // ---------------------------------------------------------------
  logic                     oreg_valid [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic [DATA_WIDTH-1:0]    oreg_data  [0:NUM_FU-1][0:MAX_FU_OUT-1];

  // Drain signals: set when the output register is consumed this cycle
  logic                     oreg_drain [0:NUM_FU-1][0:MAX_FU_OUT-1];

  // Output register sequential logic
  always_ff @(posedge clk or negedge rst_n) begin : oreg_seq
    integer iter_var0;
    integer iter_var1;
    if (!rst_n) begin : oreg_reset
      for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : rst_fu
        for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : rst_out
          oreg_valid[iter_var0][iter_var1] <= 1'b0;
          oreg_data[iter_var0][iter_var1]  <= '0;
        end : rst_out
      end : rst_fu
    end : oreg_reset
    else begin : oreg_op
      for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : upd_fu
        for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : upd_out
          if (oreg_drain[iter_var0][iter_var1]) begin : drain_oreg
            oreg_valid[iter_var0][iter_var1] <= 1'b0;
          end : drain_oreg
          if (pipe_valid[iter_var0][iter_var1] && !oreg_valid[iter_var0][iter_var1]) begin : capture_oreg
            oreg_valid[iter_var0][iter_var1] <= 1'b1;
            oreg_data[iter_var0][iter_var1]  <= pipe_data[iter_var0][iter_var1];
          end : capture_oreg
        end : upd_out
      end : upd_fu
    end : oreg_op
  end : oreg_seq

  // Status output
  always_comb begin : oreg_status
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : stat_fu
      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : stat_out
        out_reg_occupied[iter_var0][iter_var1] = oreg_valid[iter_var0][iter_var1];
      end : stat_out
    end : stat_fu
  end : oreg_status

  // ---------------------------------------------------------------
  // Register writeback: handle results marked is_reg
  //
  // For each FU output register that is valid and configured as
  // register-target (is_reg), drain it immediately and route to
  // the register writeback port.
  // ---------------------------------------------------------------
  always_comb begin : reg_writeback
    integer iter_var0;
    integer iter_var1;

    // Default: no register writes
    reg_wr_enable = '0;
    for (iter_var0 = 0; iter_var0 < MAX_FU_OUT; iter_var0 = iter_var0 + 1) begin : def_wr
      reg_wr_idx[iter_var0]  = '0;
      reg_wr_data[iter_var0] = '0;
    end : def_wr

    // Scan FUs: for each output, if it targets a register, write it
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : rw_fu
      if (fu_active[iter_var0]) begin : rw_active
        for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : rw_out
          if (oreg_valid[iter_var0][iter_var1] &&
              fu_result_is_reg[iter_var0][iter_var1]) begin : rw_is_reg
            // Use iter_var1 as write-port index (one port per FU output position)
            if (!reg_wr_enable[iter_var1]) begin : first_writer
              reg_wr_enable[iter_var1] = 1'b1;
              reg_wr_idx[iter_var1]    = fu_result_reg_idx[iter_var0][iter_var1];
              reg_wr_data[iter_var1]   = oreg_data[iter_var0][iter_var1];
            end : first_writer
          end : rw_is_reg
        end : rw_out
      end : rw_active
    end : rw_fu
  end : reg_writeback

  // ---------------------------------------------------------------
  // Per-PE-output round-robin arbitration
  // ---------------------------------------------------------------

  // Per PE output: request vector from all FUs
  logic [NUM_FU-1:0] arb_req     [NUM_PE_OUT];
  logic [NUM_FU-1:0] arb_grant   [NUM_PE_OUT];
  logic               arb_valid   [NUM_PE_OUT];
  logic [FU_IDX_W-1:0] arb_idx   [NUM_PE_OUT];

  // Build request vectors
  always_comb begin : build_arb_req
    integer iter_var0;
    integer iter_var1;
    integer iter_var2;

    for (iter_var0 = 0; iter_var0 < NUM_PE_OUT; iter_var0 = iter_var0 + 1) begin : per_pe_out
      arb_req[iter_var0] = '0;

      for (iter_var1 = 0; iter_var1 < NUM_FU; iter_var1 = iter_var1 + 1) begin : scan_fu
        if (fu_active[iter_var1]) begin : fu_act
          for (iter_var2 = 0; iter_var2 < MAX_FU_OUT; iter_var2 = iter_var2 + 1) begin : scan_out
            if (oreg_valid[iter_var1][iter_var2] &&
                !fu_out_demux_disconnect[iter_var1][iter_var2] &&
                !fu_out_demux_discard[iter_var1][iter_var2] &&
                !fu_result_is_reg[iter_var1][iter_var2] &&
                (EFF_OUT_SEL'(iter_var0) == fu_out_demux_sel[iter_var1][iter_var2])) begin : req_match
              arb_req[iter_var0][iter_var1] = 1'b1;
            end : req_match
          end : scan_out
        end : fu_act
      end : scan_fu
    end : per_pe_out
  end : build_arb_req

  // Instantiate round-robin arbiters
  genvar g_out;
  generate
    for (g_out = 0; g_out < NUM_PE_OUT; g_out = g_out + 1) begin : gen_arb

      // Ack signal: grant is consumed when PE output transfers
      logic arb_ack;
      assign arb_ack = arb_valid[g_out] & pe_out_ready[g_out];

      fabric_rr_arbiter #(
        .NUM_REQ (NUM_FU)
      ) u_rr_arb (
        .clk         (clk),
        .rst_n       (rst_n),
        .req         (arb_req[g_out]),
        .ack         (arb_ack),
        .grant       (arb_grant[g_out]),
        .grant_valid (arb_valid[g_out]),
        .grant_idx   (arb_idx[g_out])
      );

    end : gen_arb
  endgenerate

  // ---------------------------------------------------------------
  // PE output drive and drain logic
  // ---------------------------------------------------------------
  always_comb begin : pe_out_drive
    integer iter_var0;
    integer iter_var1;
    integer iter_var2;

    // Default: no PE output valid
    pe_out_valid = '0;
    for (iter_var0 = 0; iter_var0 < NUM_PE_OUT; iter_var0 = iter_var0 + 1) begin : def_pe_out
      pe_out_data[iter_var0] = '0;
      pe_out_tag[iter_var0]  = '0;
    end : def_pe_out

    // Default: no drains
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : def_drain
      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : def_drain_out
        oreg_drain[iter_var0][iter_var1] = 1'b0;
      end : def_drain_out
    end : def_drain

    // Drive PE outputs from arbiter winners
    for (iter_var0 = 0; iter_var0 < NUM_PE_OUT; iter_var0 = iter_var0 + 1) begin : drive_pe_out
      if (arb_valid[iter_var0]) begin : has_winner
        pe_out_valid[iter_var0] = 1'b1;

        // Find which FU output drives this PE output
        for (iter_var1 = 0; iter_var1 < NUM_FU; iter_var1 = iter_var1 + 1) begin : find_fu
          if (arb_grant[iter_var0][iter_var1]) begin : granted_fu
            for (iter_var2 = 0; iter_var2 < MAX_FU_OUT; iter_var2 = iter_var2 + 1) begin : find_out
              if (oreg_valid[iter_var1][iter_var2] &&
                  !fu_out_demux_disconnect[iter_var1][iter_var2] &&
                  !fu_out_demux_discard[iter_var1][iter_var2] &&
                  !fu_result_is_reg[iter_var1][iter_var2] &&
                  (EFF_OUT_SEL'(iter_var0) == fu_out_demux_sel[iter_var1][iter_var2])) begin : out_match
                pe_out_data[iter_var0] = oreg_data[iter_var1][iter_var2];
                pe_out_tag[iter_var0]  = fu_result_tag[iter_var1][iter_var2];

                // Drain on transfer
                if (pe_out_ready[iter_var0]) begin : do_drain
                  oreg_drain[iter_var1][iter_var2] = 1'b1;
                end : do_drain
              end : out_match
            end : find_out
          end : granted_fu
        end : find_fu
      end : has_winner
    end : drive_pe_out

    // Drain discard outputs immediately (no PE output needed)
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : discard_drain
      if (fu_active[iter_var0]) begin : discard_fu_act
        for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : discard_out
          if (oreg_valid[iter_var0][iter_var1] &&
              fu_out_demux_discard[iter_var0][iter_var1] &&
              !fu_out_demux_disconnect[iter_var0][iter_var1] &&
              !fu_result_is_reg[iter_var0][iter_var1]) begin : is_discard
            oreg_drain[iter_var0][iter_var1] = 1'b1;
          end : is_discard
        end : discard_out
      end : discard_fu_act
    end : discard_drain

    // Drain register-targeted outputs immediately (handled by reg writeback)
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : reg_drain
      if (fu_active[iter_var0]) begin : reg_fu_act
        for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : reg_out
          if (oreg_valid[iter_var0][iter_var1] &&
              fu_result_is_reg[iter_var0][iter_var1]) begin : is_reg_drain
            // Only drain if the register write was actually accepted
            if (reg_wr_enable[iter_var1]) begin : reg_accepted
              oreg_drain[iter_var0][iter_var1] = 1'b1;
            end : reg_accepted
          end : is_reg_drain
        end : reg_out
      end : reg_fu_act
    end : reg_drain

  end : pe_out_drive

endmodule : fabric_temporal_pe_output_arb
